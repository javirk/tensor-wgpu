use crate::{Tensor, Tensor2};
use ndarray::{prelude::*, StrideShape};

#[test]
fn zeros_creation() {
    let shape = (2, 3).f();
    let a = Tensor::<f32, _>::zeros(shape);
    assert_eq!(a.shape(), &[2, 3]);
    assert_eq!(a.data, array![[0., 0., 0.], [0., 0., 0.]]);
}

#[test]
fn from_data_creation() {
    let shape = StrideShape::from((2, 3));
    let a = Tensor::<f32, _>::from_data(vec![1., 2., 3., 4., 5., 6.], shape);
    assert_eq!(a.shape(), &[2, 3]);
    assert_eq!(a.data, array![[1., 2., 3.], [4., 5., 6.]]);
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
    let shape = StrideShape::from((2, 3));
    let a = Tensor::<f32, _>::zeros((2,3).f());
    assert_eq!(a.to_array(), vec![0., 0., 0., 0., 0., 0.]);
    let a = Tensor::<f32, _>::from_data(vec![1., 2., 3., 4., 5., 6.], shape);
    assert_eq!(a.to_array(), vec![1., 2., 3., 4., 5., 6.]);
    let shape = StrideShape::from((2, 3, 4));
    let a = Tensor::<f32, _>::from_data(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.], shape);
    assert_eq!(a.to_array(), vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.]);
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

#[test]
fn copy_dimension() {
    let shape = (2, 3).f();
    let mut a = Tensor::<f32, _>::zeros(shape);
    a[[0, 0]] = 1.;
    a.copy_dimension(0);
    assert_eq!(a.shape(), &[3, 3]);
    assert_eq!(a.data, array![[1., 0., 0.], [0., 0., 0.], [1., 0., 0.]]);

    let mut a = Tensor::<f32, _>::zeros(shape);
    a[[0, 0]] = 1.;
    a.copy_dimension(1);
    assert_eq!(a.shape(), &[2, 4]);
    assert_eq!(a.data, array![[1., 0., 0., 1.], [0., 0., 0., 0.]]);
}

#[test]
fn concatenate_vector() {
    let shape = (2, 3).f();
    let mut a = Tensor::<f32, _>::zeros(shape);
    a[[0, 0]] = 1.;
    let b = vec![2., 2., 0.];
    a.concatenate_vector(&b, 0);
    assert_eq!(a.shape(), &[3, 3]);
    assert_eq!(a.data, array![[1., 0., 0.], [0., 0., 0.], [2., 2., 0.]]);
}