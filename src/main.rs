use tch;
use tch::Tensor;

fn main() {
    if tch::Cuda::is_available() {
        println!("CUDA is available!");

        let device_count = tch::Cuda::device_count();
        println!("Device count = {device_count}");
    
        if tch::Cuda::cudnn_is_available() {
            println!("CUDNN is available!");
        }
    }

    let t = Tensor::of_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    let t = t.to(tch::Device::Cuda(0));
    t.print();
}
