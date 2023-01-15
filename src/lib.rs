use wgpu::{util::DeviceExt};
use ndarray::{prelude::*, Shape, Array, NdIndex};
use num_traits;
use std::ops::{Index, IndexMut};

pub struct Tensor<T, D> {
    pub data: Array<T, D>,
    pub buffer: Option<wgpu::Buffer>,
    pub buf_size: Option<wgpu::BufferSize>,
}

// Type aliases
pub type Tensor1<T> = Tensor<T, Ix1>;
pub type Tensor2<T> = Tensor<T, Ix2>;
pub type Tensor3<T> = Tensor<T, Ix3>;

// Initializers
impl<T, D> Tensor<T, D> where
    T: bytemuck::Pod + bytemuck::Zeroable + num_traits::identities::Zero,
    D: ndarray::Dimension,
{  // Not sure about this
    pub fn zeros(size: Shape<D>) -> Self {
        let data = Array::<T, _>::zeros(size);
        Tensor { 
            data: data, 
            buffer: None, 
            buf_size: None 
        }
    }
    // From data
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
    pub fn concatenate(&mut self, other: &mut Tensor<T, D>, dim: usize) {
        self.data = ndarray::concatenate(ndarray::Axis(dim), &[self.data.view(), other.data.view()]).unwrap();        
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

// GPU methods
impl<T, D> Tensor<T, D> where 
    T: bytemuck::Pod + bytemuck::Zeroable,
    D: ndarray::Dimension,
{
    pub fn to_array(&self) -> Vec<T> {
        Array::from_iter(self.data.iter().cloned()).to_vec()
    }

    pub fn create_buffer(&mut self, device: &wgpu::Device, usage: wgpu::BufferUsages, label: Option<&str>) {
        let buf_size = self.data.len() * std::mem::size_of::<T>();
        self.buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label,
            contents: bytemuck::cast_slice(&self.to_array()),
            usage: usage,
        }));
        self.buf_size = wgpu::BufferSize::new(buf_size as _,);
    }

    pub fn binding_resource(&self) -> wgpu::BindingResource {
        self.buffer.as_ref().expect("").as_entire_binding()
    }
}

#[cfg(test)]
mod tests;