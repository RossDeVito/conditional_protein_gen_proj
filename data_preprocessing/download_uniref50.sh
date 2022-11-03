#!/bin/sh

# Download UniRef50 XML from UniProt
wget -P ../data/ https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.xml.gz

# Download XML schema
wget -P ../data/ https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref.xsd

# Unzip the file
gzip -d ../data/uniref50.xml.gz
