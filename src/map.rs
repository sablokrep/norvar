use crate::fasta::Alignment;
use crate::fasta::FASTA;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
codeprog@icloud.com

MT_orang        16499   61      16018   +       MT_human        16569   637     16562   3196    15967
*/

pub fn alignmentmap(
    pathfile: &str,
    referencespecies: &str,
    alignmentfile: &str,
) -> Result<Vec<String>, Box<dyn Error>> {
    let fileopen = File::open(pathfile).expect("file not present");
    let fileread = BufReader::new(fileopen);
    let mut filevec: Vec<FASTA> = Vec::new();
    let mut id: Vec<String> = Vec::new();
    let mut seq: Vec<String> = Vec::new();

    for i in fileread.lines() {
        let line = i.expect("line not present");
        if line.starts_with(">") {
            id.push(line);
        } else if !line.starts_with(">") {
            seq.push(line);
        } else {
            continue;
        }
    }

    for i in 0..id.len() {
        filevec.push(FASTA {
            id: id[i].clone(),
            name: seq[i].clone(),
        })
    }

    let alignmentopen = File::open(alignmentfile).expect("file not present");
    let alignmentread = BufReader::new(alignmentopen);

    let mut alignmentstuct: Vec<Alignment> = Vec::new();
    for i in alignmentread.lines() {
        let line = i.expect("line not present");
        let linevec = line.split("\t").collect::<Vec<_>>();
        alignmentstuct.push(Alignment {
            refname: linevec[0].to_string(),
            refstart: linevec[1].parse::<usize>().unwrap(),
            refend: linevec[1].parse::<usize>().unwrap() + linevec[2].parse::<usize>().unwrap(),
        })
    }

    let mut splicevector: Vec<String> = Vec::new();
    for i in alignmentstuct.iter() {
        for val in filevec.iter() {
            if i.refname == val.id && i.refname == referencespecies {
                splicevector.push(val.name[i.refstart..i.refend].to_string());
            }
        }
    }

    Ok(splicevector)
}
