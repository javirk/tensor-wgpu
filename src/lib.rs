use wgpu::{util::DeviceExt};
use ndarray::{prelude::*, Shape, Array, NdIndex, OwnedRepr, RawDataClone, SliceArg, StrideShape};
use num_traits;
use std::ops::{Index, IndexMut};
use core::fmt;

pub struct Tensor<T, D> {
    pub data: Array<T, D>,
    pub buffer: Option<wgpu::Buffer>,
    pub buf_size: usize,
}

// Type aliases
pub type Tensor1<T> = Tensor<T, Ix1>;
pub type Tensor2<T> = Tensor<T, Ix2>;
pub type Tensor3<T> = Tensor<T, Ix3>;
pub type Tensor4<T> = Tensor<T, Ix4>;

// Initializers
impl<T, D> Tensor<T, D> where
    T: bytemuck::Pod + bytemuck::Zeroable + num_traits::identities::Zero,
    D: ndarray::Dimension,
{
    pub fn zeros(size: Shape<D>) -> Self {
        let data = Array::<T, _>::zeros(size);
        let buf_size = data.len() * std::mem::size_of::<T>();
        Tensor { 
            data: data, 
            buffer: None, 
            buf_size: buf_size
        }
    }
    // TODO: Initialize from data

}
impl<T, D> Tensor<T, D> where
    T: bytemuck::Pod + bytemuck::Zeroable,
    D: ndarray::Dimension,
{
    //pub fn from_data(data: Vec<T>, size: Shape<D>) -> Self {
    pub fn from_data(data: Vec<T>, size: StrideShape<D>) -> Self {   
        let data = Array::<T, _>::from_shape_vec(size, data).unwrap();
        let buf_size = data.len() * std::mem::size_of::<T>();
        Tensor { 
            data: data, 
            buffer: None, 
            buf_size: buf_size
        }
    }
}

// Other methods
impl<T, D> Tensor<T, D> where 
    T: bytemuck::Pod + bytemuck::Zeroable,
    D: ndarray::Dimension, 
{
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }
}

// Dimension transformations
impl<T, D> Tensor<T, D> where 
    T: bytemuck::Pod + bytemuck::Zeroable,
    D: ndarray::RemoveAxis
{
    pub fn concatenate(&mut self, other: &Tensor<T, D>, dim: usize) {
        self.data = ndarray::concatenate(ndarray::Axis(dim), &[self.data.view(), other.data.view()]).unwrap();
        self.update_buffer_size();
    }

    pub fn concatenate_vector(&mut self, vec: &Vec<T>, dim: usize) {
        let mut arr_shape = self.data.raw_dim();
        arr_shape[dim] = 1;
        let tensor = Tensor::<T, _>::from_data(vec.clone(), StrideShape::from(arr_shape));
        self.concatenate(&tensor, dim);
    }

    pub fn enlarge_dimension(&mut self, dim: usize, default_value: T) {
        let mut arr_shape = self.data.raw_dim();
        arr_shape[dim] = 1;
        let arr_concat = Array::<T,_>::from_elem(arr_shape, default_value);
        let tensor = Tensor {
            data: arr_concat,
            buffer: None,
            buf_size: 0,
        };
        self.concatenate(&tensor, dim);
    }

    pub fn copy_dimension(&mut self, dim: usize) {
        let mut tensor_copy = self.clone();
        tensor_copy.data.slice_axis_inplace(ndarray::Axis(dim), ndarray::Slice::from(0..1));
        self.concatenate(&tensor_copy, dim);
    }

    fn update_buffer_size(&mut self) {
        self.buf_size = self.data.len() * std::mem::size_of::<T>();
    }
}

// Indexing
impl<Idx: NdIndex<D>, T, D> Index<Idx> for Tensor<T, D> where
    T: bytemuck::Pod + bytemuck::Zeroable,
    D: ndarray::Dimension,
{
    type Output = <Array<T, D> as Index<Idx>>::Output;
    fn index(&self, i: Idx) -> &Self::Output {
        &self.data[i]
    }
}

impl<Idx: NdIndex<D>, T, D> IndexMut<Idx> for Tensor<T, D> where
    T: bytemuck::Pod + bytemuck::Zeroable,
    D: ndarray::Dimension,
{

    fn index_mut (&mut self, i: Idx) -> &mut Self::Output {
        &mut self.data[i]
    }
}

impl<T, D> Clone for Tensor<T, D> where OwnedRepr<T>: RawDataClone, D: Clone {
    fn clone(&self) -> Tensor<T, D> {
        Tensor {
            data: self.data.clone(),
            buffer: None,
            buf_size: self.buf_size,
        }
    }
}

// GPU methods
impl<T, D> Tensor<T, D> where 
    T: bytemuck::Pod + bytemuck::Zeroable,
    D: ndarray::Dimension,
{
    pub fn to_array(&self) -> Vec<T> {
        Array::from_iter(self.data.iter().cloned()).to_vec()
    }

    pub fn create_buffer(&mut self, device: &wgpu::Device, usage: wgpu::BufferUsages, label: Option<&str>) {
        self.buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label,
            contents: bytemuck::cast_slice(&self.to_array()),
            usage: usage,
        }));
    }

    pub fn buffer_size(&self) -> usize {
        self.buf_size
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().expect("Buffer not found")
    }

    pub fn binding_resource(&self) -> wgpu::BindingResource {
        self.buffer.as_ref().expect("").as_entire_binding()
    }
}

impl<T, D> fmt::Display for Tensor<T, D> where 
    T: fmt::Display + bytemuck::Pod + bytemuck::Zeroable + fmt::LowerExp, 
    D: ndarray::Dimension 
{

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        format!("Dimensions: {:?}\n{:+e}", self.shape(), self.data).fmt(f)
    }
}

#[cfg(test)]
mod tests;