use crate::merge_state::{CloneConverter, InPlaceVecMergeStateRef, MergeStateMut, MutateInput};
use binary_merge::MergeState;
use blake2b_simd::Params;
use fnv::FnvHashMap;
use futures::future::BoxFuture;
use futures::stream::{poll_fn, BoxStream};
use futures::{Future, FutureExt, Stream, StreamExt};
use lazy_static::lazy_static;
use parking_lot::Mutex;
use std::array::TryFromSliceError;
use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;

use super::IterKey;

enum Blob2 {
    Heap(Arc<[u8]>),
    Inline(u8, [u8; 22]),
}

type Hash = [u8; 32];
type Blob = Arc<[u8]>;
type Key = Arc<[u8]>;
type Value = Arc<[u8]>;

pub trait BlobStore: Send + Sync + std::fmt::Debug {
    fn load(&self, hash: Hash) -> BoxFuture<'_, anyhow::Result<Blob>>;
    fn store(&self, hash: Hash, value: &[u8]) -> BoxFuture<'_, anyhow::Result<()>>;
}

#[derive(Debug, Clone, Default)]
pub struct InMemBlobStore {
    data: Arc<Mutex<FnvHashMap<Hash, Blob>>>,
}

impl BlobStore for InMemBlobStore {
    fn load(&self, hash: Hash) -> BoxFuture<'_, anyhow::Result<Blob>> {
        let dict = self.data.lock();
        let res = match dict.get(&hash) {
            Some(value) => Ok(value.clone()),
            None => Err(anyhow::anyhow!("not there!")),
        };
        futures::future::ready(res).boxed()
    }

    fn store(&self, hash: Hash, value: &[u8]) -> BoxFuture<'_, anyhow::Result<()>> {
        let mut dict = self.data.lock();
        dict.insert(hash, value.into());
        futures::future::ready(Ok(())).boxed()
    }
}

lazy_static! {
    static ref EMPTY_BLOB_ARC: Arc<[u8]> = Vec::new().into();
}

lazy_static! {
    static ref EMPTY_CHILDREN_ARC: Arc<Vec<Tree>> = Vec::new().into();
}

struct Hex<'a>(&'a [u8]);

impl<'a> std::fmt::Debug for Hex<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[derive(Clone, PartialEq, Eq)]
enum Children {
    Hash(Hash),
    Data(Arc<Vec<Tree>>),
    Both(Hash, Arc<Vec<Tree>>),
}

impl std::fmt::Debug for Children {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hash(arg0) => f.debug_tuple("Hash").field(&Hex(arg0)).finish(),
            Self::Data(arg0) => f.debug_tuple("Data").field(arg0).finish(),
            Self::Both(arg0, arg1) => f.debug_tuple("Both").field(&Hex(arg0)).field(arg1).finish(),
        }
    }
}

impl Children {
    /// get the hash, or panic if there is no hash
    /// will only return None for the empty children
    fn hash(&self) -> Option<Hash> {
        match self {
            Self::Hash(hash) => Some(*hash),
            Self::Both(hash, _) => Some(*hash),
            Self::Data(data) => {
                if data.is_empty() {
                    None
                } else {
                    panic!("hash not available");
                }
            }
        }
    }

    /// Checks if data contains no elements - this is the only valid representation of empty children
    fn is_empty(&self) -> bool {
        match &self {
            Self::Data(data) => data.is_empty(),
            Self::Both(_, _) => false,
            Self::Hash(_) => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tree {
    prefix: Key,
    value: Option<Value>,
    children: Children,
}

impl Tree {
    pub fn leaf(value: Value) -> Self {
        Self {
            prefix: EMPTY_BLOB_ARC.clone(),
            value: Some(value),
            children: Children::Data(EMPTY_CHILDREN_ARC.clone()),
        }
    }

    pub fn single(key: Key, value: Value) -> Self {
        Self {
            prefix: key,
            value: Some(value),
            children: Children::Data(EMPTY_CHILDREN_ARC.clone()),
        }
    }

    pub fn prepend(&mut self, prefix: Key) {
        if !prefix.is_empty() {
            let mut t = Vec::with_capacity(self.prefix.len() + prefix.len());
            t.extend_from_slice(&self.prefix);
            t.extend_from_slice(&prefix);
            self.prefix = t.into();
        }
    }

    fn clone_shortened(&self, n: usize) -> Self {
        assert!(n < self.prefix.len());
        Self {
            prefix: self.prefix.as_ref()[n..].into(),
            value: self.value.clone(),
            children: self.children.clone(),
        }
    }

    fn split(&mut self, n: usize) {
        assert!(n < self.prefix.len());
        let first = self.prefix[..n].into();
        let rest = self.prefix[n..].into();
        let mut split = Self {
            prefix: first,
            value: None,
            children: Children::Data(EMPTY_CHILDREN_ARC.clone()),
        };
        std::mem::swap(self, &mut split);
        let mut child = split;
        // now, self is a degenerate empty node with first being the prefix
        // child is the former self (value and children) with rest as prefix
        child.prefix = rest;
        let children = self.children_mut();
        children.push(child);
        children.sort_by_key(|x| x.prefix[0]);
    }

    fn unsplit(&mut self) {
        // remove all empty children
        // this might sometimes not be necessary, but it is tricky to find out when.
        if !self.children.is_empty() {
            self.children_mut().retain(|x| !x.is_empty());
            // a single child and no own value is degenerate
            if self.children().len() == 1 && self.value.is_none() {
                let mut child = self.children_mut().pop().unwrap();
                child.prepend0(&self.prefix);
                *self = child;
            }
        }
        // canonicalize prefix for empty node
        // this might sometimes not be necessary, but it is tricky to find out when.
        if self.is_empty() {
            self.prefix = EMPTY_BLOB_ARC.clone();
        }
    }

    pub fn empty() -> Self {
        Self {
            prefix: EMPTY_BLOB_ARC.clone(),
            value: None,
            children: Children::Data(EMPTY_CHILDREN_ARC.clone()),
        }
    }

    /// True if the tree is empty
    pub fn is_empty(&self) -> bool {
        self.children.is_empty() && self.value.is_none()
    }

    pub fn union_with<'a>(
        &'a mut self,
        that: &'a Tree,
        store: &'a TreeReader,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
        self.outer_combine_with(that, store, |_, b| Some(b.clone()))
    }

    fn outer_combine_with<'a>(
        &'a mut self,
        that: &'a Tree,
        store: &'a TreeReader,
        f: impl Fn(&Value, &Value) -> Option<Value> + Copy + Send + Sync + 'static,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
        async move {
            let n = common_prefix(&self.prefix, &that.prefix);
            if n == self.prefix.len() && n == that.prefix.len() {
                // prefixes are identical
                if let Some(w) = &that.value {
                    if let Some(v) = &mut self.value {
                        self.value = f(v, w);
                    } else {
                        self.value = Some(w.clone())
                    }
                }
                let mut that = that.clone();
                that.ensure_data(&store).await?;
                self.ensure_data(&store).await?;
                self.outer_combine_children_with(that.children(), &store, f)
                    .await?;
            } else if n == self.prefix.len() {
                // self is a prefix of that
                let mut that = that.clone_shortened(n);
                that.ensure_data(&store).await?;
                self.ensure_data(&store).await?;
                self.outer_combine_children_with(&mut [that], &store, f)
                    .await?;
            } else if n == that.prefix.len() {
                // that is a prefix of self
                // split at the offset, then merge in that
                // we must not swap sides!
                self.split(n);
                // self now has the same prefix as that, so just repeat the code
                // from where prefixes are identical
                if let Some(w) = &that.value {
                    if let Some(v) = &mut self.value {
                        self.value = f(v, w);
                    } else {
                        self.value = Some(w.clone())
                    }
                }
                let mut that = that.clone();
                that.ensure_data(&store).await?;
                self.ensure_data(&store).await?;
                self.outer_combine_children_with(that.children(), &store, f)
                    .await?;
            } else {
                // disjoint
                self.split(n);
                let children = self.children_mut();
                children.push(that.clone_shortened(n));
                children.sort_by_key(|x| x.prefix[0]);
            }
            self.ensure_data(&store).await?;
            self.unsplit();
            Ok(())
        }
        .boxed()
    }

    async fn outer_combine_children_with(
        &mut self,
        rhs: &[Tree],
        store: &TreeReader,
        f: impl Fn(&Value, &Value) -> Option<Value> + Copy + Send + Sync + 'static,
    ) -> anyhow::Result<()> {
        let state = InPlaceVecMergeStateRef::new(self.children_mut(), &rhs);
        OuterCombineOp(store, f).tape_merge(state).await?;
        Ok(())
    }

    async fn ensure_hash0(&mut self, store: TreeWriter) -> anyhow::Result<()> {
        if let Children::Data(data) = &mut self.children {
            if !data.is_empty() {
                // we must recursively call ensure_hash
                for child in Arc::make_mut(data) {
                    child.ensure_hash(&store).await?;
                }
                if let Some(hash) = store.store(data.as_ref()).await? {
                    self.children = Children::Both(hash, data.clone());
                }
            }
        }
        Ok(())
    }

    pub fn ensure_hash(&mut self, store: &TreeWriter) -> BoxFuture<'_, anyhow::Result<()>> {
        self.ensure_hash0(store.clone()).boxed()
    }

    pub async fn ensure_data(&mut self, store: &TreeReader) -> anyhow::Result<()> {
        if let Children::Hash(hash) = &self.children {
            let data = store.load(*hash).await?;
            self.children = Children::Both(*hash, data);
        }
        Ok(())
    }

    pub async fn shrink(&mut self, store: &TreeWriter) -> anyhow::Result<()> {
        self.ensure_hash(store).await?;
        if let Children::Both(hash, _) = &self.children {
            self.children = Children::Hash(*hash);
        }
        Ok(())
    }

    pub fn stream<'a>(
        &self,
        store: &'a TreeReader,
    ) -> impl Stream<Item = anyhow::Result<(IterKey<u8>, Blob)>> + 'a {
        let prefix = IterKey(Arc::new(self.prefix.as_ref().into()));
        EntryStream::new(self.clone(), prefix, store).as_stream()
    }

    fn prepend0(&mut self, prefix: &[u8]) {
        if !prefix.is_empty() && !self.is_empty() {
            let mut prefix1 = Vec::new();
            prefix1.extend_from_slice(prefix);
            prefix1.extend_from_slice(&self.prefix);
            self.prefix = prefix1.into();
        }
    }

    fn to_bytes(children: &[Tree]) -> anyhow::Result<Vec<u8>> {
        let mut res = Vec::new();
        let len = children.len();
        res.extend_from_slice(&u16::to_be_bytes(u16::try_from(len)?));
        for child in children {
            let prefix_len = u16::try_from(child.prefix.len())?;
            let prefix = child.prefix.as_ref();
            res.extend_from_slice(&prefix_len.to_be_bytes());
            res.extend_from_slice(prefix);
        }
        for child in children {
            if let Some(value) = child.value.as_ref() {
                let value_len = u16::try_from(value.len())?;
                // marker for value present
                res.extend_from_slice(&[1]);
                res.extend_from_slice(&value_len.to_be_bytes());
                res.extend_from_slice(value.as_ref());
            } else {
                // marker for no value
                res.extend_from_slice(&[0]);
            }
        }
        for child in children {
            if let Some(hash) = &child.children.hash() {
                // marker for child hash present
                res.extend_from_slice(&[1]);
                res.extend_from_slice(hash);
            } else {
                // marker for no children
                res.extend_from_slice(&[0]);
            }
        }
        Ok(res)
    }
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Arc<Vec<Tree>>> {
        let mut remaining = bytes;
        let mut prefixes = Vec::new();
        let mut values = Vec::new();
        let mut hashes = Vec::new();
        let len = u16::from_be_bytes(*remaining.take::<2>()?);
        for _ in 0..len {
            let prefix_len = u16::from_be_bytes(*remaining.take::<2>()?);
            let prefix: &[u8] = remaining.take_n(prefix_len as usize)?;
            prefixes.push(prefix.into());
        }
        for _ in 0..len {
            values.push(match remaining.take::<1>()? {
                &[1] => {
                    let value_len = u16::from_be_bytes(*remaining.take::<2>()?);
                    let value: &[u8] = remaining.take_n(value_len as usize)?;
                    Some(value.into())
                }
                &[0] => None,
                _ => {
                    anyhow::bail!("unexpected prefix");
                }
            });
        }
        for _ in 0..len {
            hashes.push(match remaining.take::<1>()? {
                &[1] => Children::Hash(*remaining.take::<32>()?),
                &[0] => Children::Data(EMPTY_CHILDREN_ARC.clone()),
                _ => {
                    anyhow::bail!("unexpected prefix");
                }
            });
        }
        let res: Vec<_> = prefixes
            .into_iter()
            .zip(values.into_iter().zip(hashes.into_iter()))
            .map(|(prefix, (value, children))| Self {
                prefix,
                value,
                children,
            })
            .collect();
        Ok(Arc::new(res))
    }
    /// Returns immutable children
    ///
    /// Will panic if children are not loaded.
    fn children(&self) -> &[Tree] {
        match &self.children {
            Children::Data(d) => d.as_ref(),
            Children::Both(_, d) => d.as_ref(),
            Children::Hash(_) => panic!("children are not loaded!"),
        }
    }
    /// Returns mutable children, and clears the hash
    /// (assuming that children will be modified, so the hash will become invalid)
    ///
    /// Will panic if children are not loaded.
    fn children_mut(&mut self) -> &mut Vec<Tree> {
        if let Children::Both(_, d) = &self.children {
            self.children = Children::Data(d.clone());
        }
        match &mut self.children {
            Children::Data(d) => Arc::make_mut(d),
            Children::Both(_, _) => unreachable!(),
            Children::Hash(_) => panic!("children are not loaded!"),
        }
    }
}

/// An iterator over the elements (key and value) of a radix tree
///
/// A complication of this compared to an iterator for a normal collection is that the keys do
/// not acutally exist, but are constructed on demand during iteration.
pub struct EntryStream<'a> {
    path: IterKey<u8>,
    stack: Vec<(Tree, usize)>,
    store: &'a TreeReader,
}

impl<'a> EntryStream<'a> {
    fn new(tree: Tree, prefix: IterKey<u8>, store: &'a TreeReader) -> Self {
        Self {
            stack: vec![(tree, 0)],
            path: prefix,
            store,
        }
    }

    fn tree(&self) -> &Tree {
        &self.stack.last().unwrap().0
    }

    fn inc(&mut self) -> Option<usize> {
        let pos = &mut self.stack.last_mut().unwrap().1;
        let res = if *pos == 0 { None } else { Some(*pos - 1) };
        *pos += 1;
        res
    }

    async fn next(mut self) -> Option<(anyhow::Result<(IterKey<u8>, Blob)>, EntryStream<'a>)> {
        while !self.stack.is_empty() {
            if let Some(pos) = self.inc() {
                if let Err(e) = self
                    .stack
                    .last_mut()
                    .unwrap()
                    .0
                    .ensure_data(&self.store)
                    .await
                {
                    return Some((Err(e), self));
                }
                if pos < self.tree().children().len() {
                    let child = self.tree().children()[pos].clone();
                    self.path.append(child.prefix.as_ref());
                    self.stack.push((child.clone(), 0));
                } else {
                    self.path.pop(self.tree().prefix.len());
                    self.stack.pop();
                }
            } else if let Some(value) = self.tree().value.as_ref() {
                let path = self.path.clone();
                let value = value.clone();
                return Some((Ok((path, value)), self));
            }
        }
        None
    }

    fn as_stream(self) -> BoxStream<'a, anyhow::Result<(IterKey<u8>, Blob)>> {
        futures::stream::unfold(self, |x| x.next()).boxed()
    }
}

// common prefix of two slices.
fn common_prefix<'a, T: Eq>(a: &'a [T], b: &'a [T]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}

#[derive(Debug, Clone)]
pub struct TreeReader(Arc<dyn BlobStore>);

#[derive(Debug, Clone)]
pub struct TreeWriter(Arc<dyn BlobStore>);

#[derive(Debug, Clone)]
pub struct TreeStore {
    pub reader: TreeReader,
    pub writer: TreeWriter,
}

impl TreeReader {
    pub async fn load(&self, key: Hash) -> anyhow::Result<Arc<Vec<Tree>>> {
        let data = self.0.load(key).await?;
        let trees = Tree::from_bytes(data.as_ref())?;
        Ok(trees)
    }
}

impl TreeWriter {
    pub async fn store(&self, trees: &[Tree]) -> anyhow::Result<Option<Hash>> {
        if trees.is_empty() {
            return Ok(None);
        }
        let data = Tree::to_bytes(trees)?;
        let hash = Params::new()
            .hash_length(32)
            .to_state()
            .update(&data)
            .finalize();
        let hash = hash.as_bytes().try_into()?;
        self.0.store(hash, &data).await?;
        Ok(Some(hash))
    }
}

impl TreeStore {
    pub fn memory() -> Self {
        Self::new(Arc::new(InMemBlobStore::default()))
    }
    pub fn new(inner: Arc<dyn BlobStore>) -> Self {
        Self {
            reader: TreeReader(inner.clone()),
            writer: TreeWriter(inner),
        }
    }
}

struct Trees(TreeStore);

impl Trees {}

trait TakeExt {
    /// take a compile time fixed number of bytes
    fn take<const N: usize>(&mut self) -> Result<&[u8; N], TryFromSliceError>;
    /// take a dynamic number of bytes
    fn take_n(&mut self, n: usize) -> anyhow::Result<&[u8]>;
}

impl TakeExt for &[u8] {
    fn take<const N: usize>(&mut self) -> Result<&[u8; N], TryFromSliceError> {
        let res = self[..N].try_into();
        *self = &self[N..];
        res
    }
    fn take_n(&mut self, n: usize) -> anyhow::Result<&[u8]> {
        anyhow::ensure!(self.len() >= n);
        let res = &self[..n];
        *self = &self[n..];
        Ok(res)
    }
}

struct OuterCombineOp<'a, F>(&'a TreeReader, F);

impl<'a, F> OuterCombineOp<'a, F>
where
    F: Fn(&Value, &Value) -> Option<Value> + Copy + Send + Sync + 'static,
{
    fn cmp(&self, a: &Tree, b: &Tree) -> Ordering {
        a.prefix[0].cmp(&b.prefix[0])
    }
    fn from_a(
        &self,
        m: &mut InPlaceVecMergeStateRef<'a, Tree, Tree, CloneConverter>,
        n: usize,
    ) -> bool {
        m.advance_a(n, true)
    }
    fn from_b(
        &self,
        m: &mut InPlaceVecMergeStateRef<'a, Tree, Tree, CloneConverter>,
        n: usize,
    ) -> bool {
        m.advance_b(n, true)
    }
    async fn collision(
        &self,
        m: &mut InPlaceVecMergeStateRef<'a, Tree, Tree, CloneConverter>,
    ) -> anyhow::Result<bool> {
        let (a, b) = m.source_slices_mut();
        let av = &mut a[0];
        let bv = &b[0];
        av.outer_combine_with(bv, self.0, self.1).await?;
        // we have modified av in place. We are only going to take it over if it
        // is non-empty, otherwise we skip it.
        let take = !av.is_empty();
        Ok(m.advance_a(1, take) && m.advance_b(1, false))
    }
    async fn tape_merge(
        &self,
        m: InPlaceVecMergeStateRef<'a, Tree, Tree, CloneConverter>,
    ) -> anyhow::Result<bool> {
        let mut m = m;
        Ok(loop {
            if let Some(a) = m.a_slice().first() {
                if let Some(b) = m.b_slice().first() {
                    // something left in both a and b
                    if !match self.cmp(a, b) {
                        Ordering::Less => self.from_a(&mut m, 1),
                        Ordering::Equal => self.collision(&mut m).await?,
                        Ordering::Greater => self.from_b(&mut m, 1),
                    } {
                        // early bail out
                        break false;
                    }
                } else {
                    // b is empty, add the rest of a
                    let al = m.a_slice().len();
                    // end in any case
                    break m.a_slice().is_empty() || self.from_a(&mut m, al);
                }
            } else {
                // a is empty, add the rest of b
                let bl = m.b_slice().len();
                // end in any case
                break m.b_slice().is_empty() || self.from_b(&mut m, bl);
            };
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use futures::executor::block_on;
    use proptest::prelude::*;

    fn arb_tree() -> impl Strategy<Value = Tree> {
        any::<(Vec<u8>, Vec<u8>, [u8; 32])>().prop_map(|(key, value, hash)| {
            let mut res = Tree::single(key.into(), value.into());
            res.children = Children::Hash(hash);
            res
        })
    }

    impl Arbitrary for Tree {
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            arb_tree().boxed()
        }

        type Strategy = BoxedStrategy<Tree>;
    }

    fn blocking_iter<S>(stream: S) -> impl Iterator<Item = S::Item>
    where
        S: Stream + Unpin,
    {
        StreamIter(stream)
    }

    struct StreamIter<S>(S);

    impl<S: Stream + Unpin> Iterator for StreamIter<S> {
        type Item = S::Item;

        fn next(&mut self) -> Option<Self::Item> {
            block_on(futures::future::poll_fn(|context| {
                self.0.poll_next_unpin(context)
            }))
        }
    }

    fn entry(key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) -> Tree {
        let key: Key = key.as_ref().into();
        let value: Value = value.as_ref().into();
        Tree::single(key, value)
    }

    fn union_all(entries: impl IntoIterator<Item = Tree>) -> Tree {
        let mut union = Tree::empty();
        let store = TreeStore::memory();
        for tree in entries {
            block_on(union.union_with(&tree, &store.reader)).unwrap();
        }
        union
    }

    fn to_std_map(tree: &Tree, store: &TreeReader) -> BTreeMap<Vec<u8>, Vec<u8>> {
        blocking_iter(tree.stream(store))
            .map(|x| x.unwrap())
            .map(|(k, v)| (k.as_ref().to_vec(), v.as_ref().to_vec()))
            .collect::<BTreeMap<_, _>>()
    }

    #[test]
    fn smoke() -> anyhow::Result<()> {
        let mut a = entry(b"a", b"1");
        let b = entry(b"b", b"2");
        let c = entry(b"b", b"2");
        let store = TreeStore::memory();
        block_on(a.union_with(&b, &store.reader))?;
        println!("{:?}", a);
        block_on(a.shrink(&store.writer))?;
        println!("{:?}", store);
        block_on(a.union_with(&c, &store.reader))?;
        for (k, v) in blocking_iter(a.stream(&store.reader).map(|x| x.unwrap())) {
            println!(
                "{} {}",
                std::str::from_utf8(&k).unwrap(),
                std::str::from_utf8(&v).unwrap()
            );
        }
        Ok(())
    }

    #[test]
    fn build_roundtrip_1() {
        let store = TreeStore::memory();
        let res = union_all([entry(&[0], &[])]);
        let actual = blocking_iter(res.stream(&store.reader))
            .map(|x| x.unwrap())
            .map(|(k, v)| (k.as_ref().to_vec(), v.as_ref().to_vec()))
            .collect::<BTreeMap<_, _>>();
        println!("{:?}", res);
        println!("{:?}", actual);
    }

    #[test]
    fn build_shrink_roundtrip_1() {
        let store = TreeStore::memory();
        let mut res = union_all([entry(&[], &[]), entry(&[66], &[]), entry(&[66, 0], &[])]);
        println!("before shrink");
        println!("{:?}", res);
        for (k, v) in blocking_iter(res.stream(&store.reader)).map(|x| x.unwrap()) {
            println!("{:?} {:?}", k, v);
        }
        block_on(res.ensure_hash(&store.writer)).unwrap();
        println!("after ensure_hash");
        println!("{:?}", res);
        block_on(res.shrink(&store.writer)).unwrap();
        println!("after shrink");
        println!("{:?}", res);
        block_on(res.ensure_data(&store.reader)).unwrap();
        println!("after ensure_data");
        println!("{:?}", res);
        for (k, v) in blocking_iter(res.stream(&store.reader)).map(Result::unwrap) {
            println!("{:?} {:?}", k, v);
        }
        let m = to_std_map(&res, &store.reader);
        println!("{:?}", m);
    }

    proptest! {
        #[test]
        fn blob_roundtrip(trees in any::<Vec<Tree>>()) {
            let blob = Tree::to_bytes(&trees).unwrap();
            let trees1 = Arc::try_unwrap(Tree::from_bytes(&blob).unwrap()).unwrap();
            prop_assert_eq!(trees, trees1);
        }

        /// build a tree from a map, and make sure it acually contains the right elements
        #[test]
        fn build_roundtrip(expected in any::<BTreeMap<Vec<u8>, Vec<u8>>>()) {
            let store = TreeStore::memory();
            let entries = expected.iter().map(|(k, v)| entry(&k, &v));
            let res = union_all(entries);
            let actual = to_std_map(&res, &store.reader);
            prop_assert_eq!(expected, actual);
        }

        /// build a tree from a map, and make sure it acually contains the right elements
        #[test]
        fn build_shrink_roundtrip(expected in any::<BTreeMap<Vec<u8>, Vec<u8>>>()) {
            let store = TreeStore::memory();
            let entries = expected.iter().map(|(k, v)| entry(&k, &v));
            let mut res = union_all(entries);
            block_on(res.shrink(&store.writer)).unwrap();
            let actual = to_std_map(&res, &store.reader);
            prop_assert_eq!(expected, actual);
        }
    }
}
