# Title     : GGPlot Graphs for Training Results
# Created by: Jack
# Created on: 5/16/20

# Add libraries
library(here) # Finding current directory
library(ggplot2) # Creating visualizations

files <- list.files(paste(here(), "/training results", sep = "", collapse = NULL), full.names = TRUE)
files
data <- lapply(files, read.csv)
data <- data[-c(8)] # Removes the results_readme.txt file
data <- data[-c(6)] # Removes the ggplot_graphs.r file

plots <- lapply(data, function(df) {
  ggplot(data = df, aes(x = ))
})