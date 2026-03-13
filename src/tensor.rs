use crate::map::alignmentmap;
use burn::prelude::Backend;
use burn::tensor::Tensor;

/*
Gaurav Sablok
codeprog@icloud.com
*/

pub fn tensor_autoencoder<B: Backend>(
    path1: &str,
    path2: &str,
    path3: &str,
    seqlen: &str,
) -> Tensor<B, 3> {
    let fastaunpack = alignmentmap(path1, path2, path3).unwrap();
    let stringcheck = fastaunpack
        .iter()
        .map(|x| x.to_uppercase())
        .collect::<Vec<_>>();
    let mut tensorveca: Vec<Tensor<B, 1>> = Vec::new();

    for i in stringcheck.iter() {
        let stringvec = i.chars().collect::<Vec<_>>();
        let mut charvec: Vec<Vec<f64>> = Vec::new();
        for val in stringvec.iter() {
            match val {
                'A' => charvec.push(vec![1.0, 0.0, 0.0, 0.0]),
                'T' => charvec.push(vec![0.0, 1.0, 0.0, 0.0]),
                'G' => charvec.push(vec![0.0, 0.0, 1.0, 0.0]),
                'C' => charvec.push(vec![0.0, 0.0, 0.0, 1.0]),
                _ => continue,
            }
        }
        let tensorvec = Tensor::<B, 1>::from_data(
            charvec
                .iter()
                .cloned()
                .flatten()
                .collect::<Vec<_>>()
                .as_slice(),
            &B::Device::default(),
        );
        tensorveca.push(tensorvec);
    }

    let finaltensor: Tensor<B, 3> =
        Tensor::stack::<3>(tensorveca, 0).reshape([1, 4, seqlen.parse::<usize>().unwrap()]);

    finaltensor
}
