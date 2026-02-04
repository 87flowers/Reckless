use std::cell::UnsafeCell;

#[repr(transparent)]
pub struct SyncUnsafeCell<T: ?Sized> {
    value: UnsafeCell<T>,
}

unsafe impl<T: ?Sized + Sync> Sync for SyncUnsafeCell<T> {}

impl<T> SyncUnsafeCell<T> {
    pub const fn new(value: T) -> Self {
        Self { value: UnsafeCell::new(value) }
    }

    pub const fn get(&self) -> *mut T {
        self.value.get()
    }
}
