use crate::Tensor;
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