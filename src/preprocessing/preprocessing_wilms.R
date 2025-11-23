#!/usr/bin/env Rscript

## ============================================================
## Script: normalize_GSE53224_Wegert.R
## Purpose:
##   - Download CEL files for GSE53224 (Wegert et al.)
##   - Perform RMA normalization (Affymetrix HG-U133A)
##   - Annotate probes with gene symbols (hgu133a.db)
##   - Collapse multiple probes per gene using WGCNA::collapseRows
##   - Write gene-level normalized expression matrix to CSV
## ============================================================

## -------------------------
## 1. Global options
## -------------------------
options(stringsAsFactors = FALSE)

## -------------------------
## 2. Package loading
##    (Assumes packages are already installed; see README)
## -------------------------
required_pkgs <- c(
  "BiocManager",  # for installation (if needed)
  "affy",
  "GEOquery",
  "hgu133a.db",
  "WGCNA"
)

for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(
      sprintf(
        "Package '%s' is not installed. Please install it via BiocManager::install('%s') before running this script.",
        pkg, pkg
      )
    )
  }
}

library(affy)
library(GEOquery)
library(hgu133a.db)
library(WGCNA)

## -------------------------
## 3. Parameters / paths
## -------------------------
gse_id      <- "GSE53224"
base_dir    <- "CEL_DEU"
raw_dir     <- file.path(base_dir, gse_id)
cel_dir     <- file.path(base_dir, "all_data")
output_dir  <- "output"

if (!dir.exists(base_dir))   dir.create(base_dir, recursive = TRUE)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

## -------------------------
## 4. Download & extract CEL files
## -------------------------

# Download supplementary files only if not already present
tar_file <- file.path(raw_dir, paste0(gse_id, "_RAW.tar"))

if (!file.exists(tar_file)) {
  message("Downloading GEO supplementary files for ", gse_id, " ...")
  getGEOSuppFiles(GEO = gse_id, makeDirectory = TRUE, baseDir = base_dir)
} else {
  message("Found existing TAR file: ", tar_file)
}

# Extract CEL files
if (!dir.exists(cel_dir)) {
  dir.create(cel_dir, recursive = TRUE)
  message("Extracting CEL files to: ", cel_dir)
  untar(tar_file, exdir = cel_dir)
} else {
  message("CEL directory already exists: ", cel_dir)
}

## -------------------------
## 5. Read and normalize data (RMA)
## -------------------------
message("Reading CEL files and performing RMA normalization...")

Data_deu <- ReadAffy(celfile.path = cel_dir)

# RMA: background correction, quantile normalization,
#      summarization via median polish
eset_rma_deu <- rma(Data_deu)

# Expression matrix as data.frame (probes x samples)
norm_data_deu <- as.data.frame(exprs(eset_rma_deu))

## -------------------------
## 6. Add gene symbols (hgu133a.db)
## -------------------------
message("Annotating probes with gene symbols...")

# hgu133aSYMBOL: map probe_id -> gene symbol
annot_df <- as.data.frame(hgu133aSYMBOL)  # columns: probe_id, symbol
rownames(annot_df) <- annot_df$probe_id

# Merge annotation with expression matrix
# First bring probe IDs into a column in the expression data
norm_data_deu$probe_id <- rownames(norm_data_deu)

norm_data_withGeneSymbol_deu <- merge(
  annot_df,
  norm_data_deu,
  by = "probe_id"
)

# Optional sanity check for a specific probe:
# annot_df["1053_at", ]

# Create rownames "SYMBOL__PROBEID"
rownames(norm_data_withGeneSymbol_deu) <- paste(
  norm_data_withGeneSymbol_deu$symbol,
  norm_data_withGeneSymbol_deu$probe_id,
  sep = "__"
)

# Order by gene symbol for convenience
norm_data_withGeneSymbol_deu <- norm_data_withGeneSymbol_deu[
  order(norm_data_withGeneSymbol_deu$symbol),
]

## -------------------------
## 7. Collapse probes to gene level (WGCNA::collapseRows)
## -------------------------
message("Collapsing probes to gene level using WGCNA::collapseRows...")

# Prepare inputs for collapseRows
rowGroup_deu <- norm_data_withGeneSymbol_deu$symbol    # gene symbols
rowID_deu    <- norm_data_withGeneSymbol_deu$probe_id  # probe IDs

# Keep only expression columns (remove annotation columns)
datET_deu <- subset(
  norm_data_withGeneSymbol_deu,
  select = -c(probe_id, symbol)
)

# Row names must be probe IDs
rownames(datET_deu) <- rowID_deu

# Collapse rows: one value per gene (MaxMean)
collapseRows_object_deu <- collapseRows(
  datET    = as.matrix(datET_deu),
  rowGroup = rowGroup_deu,
  rowID    = rowID_deu,
  method   = "MaxMean"
)

# Gene-level expression matrix (genes x samples)
norm_data_deu_gene <- as.data.frame(collapseRows_object_deu$datETcollapsed)

## -------------------------
## 8. Write output
## -------------------------
date_stamp <- format(Sys.Date(), "%Y-%m-%d")
out_file   <- file.path(
  output_dir,
  paste0("norm_data_dkfz_wilms", date_stamp, ".csv")
)

message("Writing gene-level normalized data to: ", out_file)
write.csv(norm_data_deu_gene, out_file, quote = TRUE)

message("Done.")
