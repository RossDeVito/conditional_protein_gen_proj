# Download UniRef50 from UniProt
wget -P ../data/ https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz

# Unzip the file
gzip -d ../data/uniref50.fasta.gz
