<div align="center">
    <h1>
    Welcome to LineMut!
    </h1>
    <p>
     LineMut is an open-source pipeline for SNV detection and accurate lineage reconstruction.
    </p>
</div>

****

<div align="center">
    <img src='./assets/Schematic_of_LineMut_pipeline_for_GitHub.png' width=1000> 
</div>

## Install
To use LineMut, first clone this code repository:  

``` bash 
git clone https://github.com/RuiyingChenBioinfo/LineMut.git
```

Create a conda environment that contains the dependencies required to run LineMut.  

``` bash
cd linemut/
conda create -n linemut_env --file ./conda_env.txt
```

LineMut also depends on [GATK4](https://gatk.broadinstitute.org/hc/en-us), so please ensure that GATK4 is already installed. You can verify the installation using the following command:  

``` bash
gatk --version 
```

Add the cloned local directory to the PATH (optional):  

```
echo PATH=$(pwd):'${PATH}' >> ~/.bashrc 
source ~/.bashrc
```

## Usage  

### linemut-call

linemut_call is designed for detecting expressed single-nucleotide variants from single-cell RNA-sequencing or spatial transcriptomics data.

``` bash
linemut_call [OPTIONS]
```
    
```
--bam, -I:
    raw BAM file.
--ref, -R:
    The reference genome sequence file in FASTA format.
--output, -O:
    Directory for saving the result.
--barcode-celltype-mapping, -m:
    The CSV file containing cell barcodes and their corresponding cell types.

--cells-coordinate, -c:
    (optional) The CSV file containing cell coordinate information. If this
    parameter is not provided, the default behavior is to use cell type as
    the unit for mutation detection without CMB partitioning.
--known-variants-dir, -v:
    (optional) A directory of VCF-formatted known variant sites for the species.
--k-mer, -k:
    (optional) The length of k-mer, default: 9.
--cell-barcode-tag, -t:
    (optional) The tag name denoting the cell barcode in the BAM file
    defaults to 'CB'.
--python:
    (optional) The pathname to the Python interpreter you want to use.
    By default, it uses the first 'python3' found in the PATH.
--gatk:
    (optional) The pathname of the GATK executable file. By default,
    search for 'gatk' in the PATH.
--samtools:
    (optional) The pathname of the samtools. By default, search for the
    'samtools' in the PATH.

--help, -h:
    Print this message, exit and return a non-zero exit status.
```

## Tutorials

Refer to the [linemut-tutorial](https://ruiyingchenbioinfo.github.io/LineMut_tutorialv0.1.0/) page for more detailed information.


<details> <summary> Note about package development </summary>

LineMut is actively being developed. You may occasionally encounter bugs or minor documentation issues. GitHub issues are welcome for bug reports, usage questions, and feature suggestions.

</details>

