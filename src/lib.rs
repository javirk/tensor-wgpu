use wgpu::{util::DeviceExt};
use ndarray::{prelude::*, Shape};
use ndarray::{array, Array};
use num_traits;

pub struct Tensor<T, D> {
    pub data: Array<T, D>,
    pub buffer: Option<wgpu::Buffer>,
    pub buf_size: Option<wgpu::BufferSize>,
}

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

// GPU methods
impl<T, D> Tensor<T, D> where 
    T: bytemuck::Pod + bytemuck::Zeroable ,
    D: ndarray::Dimension,
{
    pub fn to_array(&self) -> Vec<T> {
        self.data.as_slice().unwrap().to_vec()
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
mod tests {
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

    // Testing GPU methods is very complicated because I need the framework.

}