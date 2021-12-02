use binary_merge::MergeState;
use blake2b_simd::Params;
use futures::future::BoxFuture;
use futures::FutureExt;
use lazy_static::lazy_static;
use std::array::TryFromSliceError;
use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use crate::merge_state::{InPlaceVecMergeStateRef, MergeStateMut, MutateInput};

type Hash = [u8; 32];
type Blob = Arc<[u8]>;
type Key = Arc<[u8]>;
type Value = Arc<[u8]>;

trait BlobStore: Send + Sync {
    fn load(&self, hash: Hash) -> BoxFuture<'static, anyhow::Result<Blob>>;
    fn store(&self, hash: Hash, value: &[u8]) -> BoxFuture<'static, anyhow::Result<()>>;
}

lazy_static! {
    static ref EMPTY_BLOB_ARC: Arc<[u8]> = Vec::new().into();
}

lazy_static! {
    static ref EMPTY_CHILDREN_ARC: Arc<Vec<Tree>> = Vec::new().into();
}

#[derive(Clone)]
enum Children {
    Hash(Hash),
    Data(Arc<Vec<Tree>>),
    Both(Hash, Arc<Vec<Tree>>),
}

impl Children {
    fn hash(&self) -> Option<Hash> {
        match self {
            Self::Hash(hash) => Some(*hash),
            Self::Both(hash, _) => Some(*hash),
            _ => None,
        }
    }
}

#[derive(Clone)]
struct Tree {
    prefix: Key,
    value: Option<Value>,
    children: Children,
}

impl Tree {
    fn leaf(value: Value) -> Self {
        Self {
            prefix: EMPTY_BLOB_ARC.clone(),
            value: Some(value),
            children: Children::Data(EMPTY_CHILDREN_ARC.clone()),
        }
    }

    fn single(key: Key, value: Value) -> Self {
        Self {
            prefix: key,
            value: Some(value),
            children: Children::Data(EMPTY_CHILDREN_ARC.clone()),
        }
    }

    fn prepend(&mut self, prefix: Key) {
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
        self.children_mut().retain(|x| !x.is_empty());
        // a single child and no own value is degenerate
        if self.children().len() == 1 && self.value.is_none() {
            let mut child = self.children_mut().pop().unwrap();
            child.prepend0(&self.prefix);
            *self = child;
        }
        // canonicalize prefix for empty node
        // this might sometimes not be necessary, but it is tricky to find out when.
        if self.is_empty() {
            self.prefix = EMPTY_BLOB_ARC.clone();
        }
    }

    /// True if the tree is empty
    fn is_empty(&self) -> bool {
        self.children().is_empty() && self.value.is_none()
    }

    fn outer_combine_with(
        &mut self,
        that: Tree,
        store: TreeStore,
        f: impl Fn(&mut Value, &[u8]) -> bool + Copy + Send + Sync + 'static,
    ) -> BoxFuture<'_, anyhow::Result<()>> {
        async move {
            let mut that = that;
            let n = common_prefix(&self.prefix, &that.prefix);
            if n == self.prefix.len() && n == that.prefix.len() {
                // prefixes are identical
                if let Some(w) = &that.value {
                    if let Some(v) = &mut self.value {
                        if !f(v, &w) {
                            self.value = None;
                        }
                    } else {
                        self.value = Some(w.clone())
                    }
                }
                that.ensure_data(&store).await?;
                self.ensure_data(&store).await?;
                self.outer_combine_children_with(that.children_mut(), &store, f)
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
                        if !f(v, &w) {
                            self.value = None;
                        }
                    } else {
                        self.value = Some(w.clone())
                    }
                }
                that.ensure_data(&store).await?;
                self.ensure_data(&store).await?;
                self.outer_combine_children_with(that.children_mut(), &store, f)
                    .await?;
            } else {
                // disjoint
                self.split(n);
                let children = self.children_mut();
                children.push(that.clone_shortened(n));
                children.sort_by_key(|x| x.prefix[0]);
            }
            self.unsplit();
            Ok(())
        }
        .boxed()
    }

    async fn outer_combine_children_with(
        &mut self,
        rhs: &mut [Tree],
        store: &TreeStore,
        f: impl Fn(&mut Value, &[u8]) -> bool + Copy + Send + Sync + 'static,
    ) -> anyhow::Result<()> {
        let state = InPlaceVecMergeStateRef::new(self.children_mut(), &rhs);
        OuterCombineOp(store, f).tape_merge(state).await?;
        Ok(())
    }

    async fn ensure_data(&mut self, store: &TreeStore) -> anyhow::Result<()> {
        if let Children::Data(data) = &self.children {
            let hash = store.store(data.as_ref()).await?;
            self.children = Children::Both(hash, data.clone());
        }
        Ok(())
    }

    async fn ensure_hash(&mut self, store: &TreeStore) -> anyhow::Result<()> {
        if let Children::Data(data) = &self.children {
            let hash = store.store(data.as_ref()).await?;
            self.children = Children::Both(hash, data.clone());
        }
        Ok(())
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

// common prefix of two slices.
fn common_prefix<'a, T: Eq>(a: &'a [T], b: &'a [T]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}

#[derive(Clone)]
struct TreeStore(Arc<dyn BlobStore>);

impl TreeStore {
    async fn load(&self, key: Hash) -> anyhow::Result<Arc<Vec<Tree>>> {
        let data = self.0.load(key).await?;
        let trees = Tree::from_bytes(data.as_ref())?;
        Ok(trees)
    }
    async fn store(&self, trees: &[Tree]) -> anyhow::Result<Hash> {
        let data = Tree::to_bytes(trees)?;
        let hash = Params::new()
            .hash_length(32)
            .to_state()
            .update(&data)
            .finalize();
        let hash = hash.as_bytes().try_into()?;
        self.0.store(hash, &data).await?;
        Ok(hash)
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

#[derive(Clone, Copy)]
struct OuterCombineOp<'a, F>(&'a TreeStore, F);

impl<'a, F> OuterCombineOp<'a, F>
where
    F: Fn(&mut Value, &[u8]) -> bool + Copy + Send + Sync + 'static,
{
    fn cmp(self, a: &Tree, b: &Tree) -> Ordering {
        a.prefix[0].cmp(&b.prefix[0])
    }
    fn from_a(self, m: &mut InPlaceVecMergeStateRef<'a, Tree, Tree>, n: usize) -> bool {
        m.advance_a(n, true)
    }
    fn from_b(self, m: &mut InPlaceVecMergeStateRef<'a, Tree, Tree>, n: usize) -> bool {
        m.advance_b(n, true)
    }
    async fn collision(
        self,
        m: &mut InPlaceVecMergeStateRef<'a, Tree, Tree>,
    ) -> anyhow::Result<bool> {
        let (a, b) = m.source_slices_mut();
        let av = &mut a[0];
        let bv = b[0].clone();
        av.outer_combine_with(bv, self.0.clone(), self.1).await?;
        // we have modified av in place. We are only going to take it over if it
        // is non-empty, otherwise we skip it.
        let take = !av.is_empty();
        Ok(m.advance_a(1, take) && m.advance_b(1, false))
    }
    async fn tape_merge(
        &self,
        m: InPlaceVecMergeStateRef<'a, Tree, Tree>,
    ) -> anyhow::Result<bool> {
        let mut m = m;
        Ok(loop {
            if let Some(a) = m.a_slice().first() {
                if let Some(b) = m.b_slice().first() {
                    let res = self.cmp(a, b);
                    // something left in both a and b
                    if !match res {
                        Ordering::Less => self.from_a(&mut m, 1),
                        Ordering::Equal => self.collision(&mut m).await?,
                        Ordering::Greater => self.from_b(&mut m, 1),
                    } {
                        break false;
                    }
                } else {
                    // b is empty, add the rest of a
                    let al = m.a_slice().len();
                    break m.a_slice().is_empty() || self.from_a(&mut m, al);
                }
            } else {
                // a is empty, add the rest of b
                let bl = m.b_slice().len();
                break m.b_slice().is_empty() || self.from_b(&mut m, bl);
            };            
        })
    }
}
