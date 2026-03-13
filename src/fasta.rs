#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct FASTA {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct Alignment {
    pub refname: String,
    pub refstart: usize,
    pub refend: usize,
}
