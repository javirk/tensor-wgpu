use wgpu::{util::DeviceExt};

pub struct Matrix<T> {
    pub matrix: Vec<T>,
    pub buffer: Option<wgpu::Buffer>,
    pub buf_size: Option<wgpu::BufferSize>,
    pub dimensions: Vec<u32>,
}

impl<T> Matrix<T> where T: bytemuck::Pod + bytemuck::Zeroable {  // Not sure about this
    pub fn add_buffer(&mut self, device: &wgpu::Device, label: Option<&str>) {
        let buf_size = self.matrix.len() * std::mem::size_of::<T>();
        self.buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label,
            contents: bytemuck::cast_slice(&self.matrix),
            usage: wgpu::BufferUsages::STORAGE,
        }));
        self.buf_size = wgpu::BufferSize::new(buf_size as _,);
    }

    // Methods: 
    // Stack
    // Increase dimension
    // Build buffer
    // Build from vector
    // Len
    // Dimensions


    pub fn add_uniform_column(&mut self, value: T) {
        for i in 0..self.num_rows {
            let index = (i + 1) * self.num_columns;
            self.matrix.insert(index as usize, value);
        }
        self.num_columns += 1;
    }

    pub fn add_row(&mut self, row: Vec<T>) {
        assert!(row.len() == self.num_columns as usize);
        self.matrix.extend(row);
        self.num_rows += 1;
    }

    pub fn matrix_to_matrix(&mut self, new_mtx: &mut Vec<T>, prev_rowcol: usize, new_rowcol: usize, third_dim: usize) {
        // Copies the matrix in self to a new matrix with a different size.
        for i in 0..prev_rowcol {
            for j in 0..prev_rowcol {
                for k in 0..third_dim {
                    new_mtx[i + j * new_rowcol + k * new_rowcol * new_rowcol] = self.matrix[i + j * prev_rowcol + k * prev_rowcol * prev_rowcol];
                }
            }
        }
    }

    // fn increase_dimension(arr: &mut [i32], dim: usize, new_size: usize) {
    //     let old_size = arr.len();
    //     let dim_size = old_size / new_size;
    //     let new_len = old_size + dim_size;
    //     let mut new_arr = vec![0; new_len];
    
    //     for i in 0..dim_size {
    //         let old_index = i * new_size;
    //         let new_index = i * (new_size + 1);
    //         new_arr[new_index..new_index + new_size].copy_from_slice(&arr[old_index..old_index + new_size]);
    //     }
    
    //     *arr = new_arr;
    // }
}