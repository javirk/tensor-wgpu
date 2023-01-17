use crate::{Tensor, Tensor2};
use ndarray::prelude::*;

#[test]
fn creation() {
    let shape = (2, 3).f();
    let a = Tensor::<f32, _>::zeros(shape);
    assert_eq!(a.shape(), &[2, 3]);
    assert_eq!(a.data, array![[0., 0., 0.], [0., 0., 0.]]);
}

#[test]
fn concatenation() {
    let shape = (2, 3).f();
    let mut a = Tensor::<f32, _>::zeros(shape);
    let mut b = Tensor::<f32, _>::zeros(shape);
    a.concatenate(&mut b, 0);
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.data, array![[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]);

    let mut a = Tensor::<f32, _>::zeros(shape);
    let mut b = Tensor::<f32, _>::zeros(shape);
    a.concatenate(&mut b, 1);
    assert_eq!(a.shape(), &[2, 6]);
    assert_eq!(a.data, array![[0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]]);
}

#[test]
fn to_array() {
    let shape = (2, 3).f();
    let a = Tensor::<f32, _>::zeros(shape);
    assert_eq!(a.to_array(), vec![0., 0., 0., 0., 0., 0.]);
}

#[test]
fn slicing() {
    let shape = (2, 3).f();
    let mut a = Tensor::<f32, _>::zeros(shape);
    a.data[[0, 0]] = 1.;
    assert_eq!(a[[0, 0]], 1.);
}

// Testing GPU methods is very complicated because I need the framework.

#[test]
fn alias_tensor2() {
    let shape = (2, 3).f();
    let a = Tensor::<f32, _>::zeros(shape);
    let b = Tensor2::<f32>::zeros(shape);
    assert_eq!(a.shape(), b.shape());
}

#[test]
fn struct_creation() {
    struct TestStruct(Tensor2<u32>);
    let val = TestStruct(Tensor2::<u32>::zeros((2, 3).f()));
    assert_eq!(val.0.shape(), &[2, 3]);
}

#[test]
fn clone() {
    let shape = (2, 3).f();
    let a = Tensor::<f32, _>::zeros(shape);
    let b = a.clone();
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.data, b.data);
    assert_eq!(a.buf_size, b.buf_size);
}

#[test]
fn enlarge_dimension() {
    let shape = (2, 3).f();
    let mut a = Tensor::<f32, _>::zeros(shape);
    a.enlarge_dimension(0, 1.);
    assert_eq!(a.shape(), &[3, 3]);
    assert_eq!(a.data, array![[0., 0., 0.], [0., 0., 0.], [1., 1., 1.]]);
}