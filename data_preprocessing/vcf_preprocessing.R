args <- commandArgs(trailingOnly = TRUE)

sample1_path<-args[1]
sample2_path<-args[2]
unimportant_path<-args[3]
file_path<-args[4]

required_packages <- c("dplyr", "readxl")

# Install missing packages
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "http://cran.rstudio.com/")
  }
}

# Install all missing packages
invisible(sapply(required_packages, install_if_missing))

library(dplyr)
library(readxl)

samp1 <- read.delim(sample1_path, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
samp2 <- read.delim(sample2_path, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
combined_vcf <- rbind(samp1, samp2)
filtered_vcf <- combined_vcf[!duplicated(combined_vcf[, c("Chromosome", "Position", "Your_DNA")]), ]

all_common_mutations <- read.csv(unimportant_path)
all_common_mutations <- unique(all_common_mutations)
sample_data <- filtered_vcf
sample_data <- sample_data[!paste(sample_data$Chromosome, sample_data$Position, sample_data$Your_DNA) %in% 
                             paste(all_common_mutations$Chromosome, all_common_mutations$Position, all_common_mutations$Your_DNA), ]
write.csv(sample_data,file_path)
