use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/*
Gaurav Sablok
codeprog@icloud.com
*/

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    encoder: Linear<B>,
    decoder: Linear<B>,
    activation: Relu,
    seqlen: usize,
}

impl<B: Backend> Encoder<B> {
    pub fn new(device: &B::Device, seqa: usize, latentdim: usize) -> Self {
        let input_dim = seqa * 4usize;
        Self {
            encoder: LinearConfig::new(input_dim, latentdim).init(device),
            decoder: LinearConfig::new(latentdim, input_dim).init(device),
            activation: Relu::new(),
            seqlen: seqa,
        }
    }

    pub fn forward(self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batchsize, channels, seqlen] = input.dims();
        let x = input.reshape([batchsize, channels * seqlen]);
        let latent = self.activation.forward(self.encoder.forward(x));
        let recontruct = self.decoder.forward(latent);
        recontruct.reshape([batchsize, 4, seqlen])
    }
    pub fn reconstruct_seq(logits: Tensor<B, 3>) -> String {
        let indices = logits.argmax(1).squeeze::<0>();
        let data: Vec<i32> = indices.into_data().convert::<i32>().to_vec().unwrap();
        data.iter()
            .map(|&idx| match idx {
                0 => "A",
                1 => "C",
                2 => "T",
                3 => "G",
                _ => "N",
            })
            .collect()
    }
}
