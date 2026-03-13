mod autoencode;
mod fasta;
mod map;
mod tensor;
use crate::autoencode::Encoder;
use crate::tensor::tensor_autoencoder;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn::tensor::Device;

/*
Gaurav Sablok
codeprog@icloud.com
*/

fn main() {
    type B = dyn Backend;
    let path1 = "";
    let path2 = "";
    let path3 = "";
    let path4 = "";
    let seqlen = path4.len();
    let latent_dim = 4usize;
    let targetdna = path4;
    let modelrun: Encoder<B> = Encoder::new(Device::default(), seqlen, latent_dim);
    let optimization = AdamConfig::new().init::<B, Encoder<B>>();
    let tensorshape = tensor_autoencoder(path1, path2, path3, path4);
    for epoch in 1..101 {
        let output = modelrun.clone().forward(tensorshape.clone());
        let loss = output
            .clone()
            .sub(tensorshape.clone())
            .powf_scalar(2.0)
            .mean();
        if epoch % 20 == 0 {
            println!("Epoch:{}, Loss:{}", epoch, loss.into_scalar());
        }
    }
    let finalconstruct = modelrun.clone().forward(tensorshape);
    let mut stringvec = Vec::new();
    let indices = finalconstruct
        .argmax(1)
        .squeeze::<0>()
        .into_data()
        .as_slice()
        .unwrap()
        .iter()
        .map(|&x| match x {
            0 => stringvec.push("A"),
            1 => stringvec.push("T"),
            2 => stringvec.push("G"),
            3 => stringvec.push("C"),
            _ => stringvec.push(""),
        });
    println!(
        "The reconstructed sequence is: {}",
        stringvec
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .concat()
    );
}
