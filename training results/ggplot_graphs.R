# Title     : GGPlot Graphs for Training Results
# Created by: Jack
# Created on: 5/16/20

# Add libraries
library(here) # Finding current directory
library(ggplot2) # Creating visualizations
library(tidyverse)

# Helper function definition
capStr <- function(y) {
  c <- strsplit(y, " ")[[1]]
  paste(toupper(substring(c, 1,1)), substring(c, 2),
      sep="", collapse=" ")
}

# Load the dataframes for each csv
files <- list.files(paste(here(), "/training results", sep = "", collapse = NULL), full.names = TRUE)
data <- lapply(files, read.csv)

# Remove non csv files
data <- data[-c(8)] # Removes the results_readme.txt file
data <- data[-c(6)] # Removes the ggplot_graphs.r file
files <- files[-c(8)] # Removes the results_readme.txt file
files <- files[-c(6)] # Removes the ggplot_graphs.r file
files <- lapply(files, function(filename) {
  return(tail(strsplit(filename,"/")[[1]], n=1))
})
files <- lapply(files, function(filename) {
  return(head(strsplit(filename,"\\.")[[1]], n=1))
})
files <- lapply(files, function(filename) {
  return(capStr(head(str_replace_all(filename,"_", " ")[[1]], n=1)))
})

# Add filenames as a column to each dataframe
data <- lapply(seq_along(data), function(index) {
  df <- data.frame("filename" = rep(files[[index]], nrow(data[[index]])))
  df <- merge(data[[index]], df)
  return(df)
})

# Combine all of the dataframes into a master df
df <- data %>% reduce(full_join)

# Mobilenet epoch validation accuracy graph
ggplot(data = df[df$filename == "Mobilenet",], aes(x = X, y = acc * 100)) +
  theme(
    panel.background = element_rect(fill = "white", color = "white",  size = 0.5, linetype = "solid"),
    panel.grid.major = element_line(size = 0.25, linetype = 'dashed', color = "black"),
    panel.grid.minor = element_line(size = 0.1, linetype = 'dotted', color = "black")
  ) +
  ggtitle("Mobilenet Model Accuracy") +
  guides( color=guide_legend(title="Data", labels = c("Training", "Validation"))) +
  xlab("Epoch") +
  ylab("Accuracy (Percent Correct)") +
  geom_line(size = 1, aes(color = "Training Data")) +
  geom_line(aes(y = val_acc * 100, color = "Test Data"), size = 1)

# Compare all of the networks' accuracies
ggplot(data = df, aes(x = X, y = acc, color=filename)) +
  theme(
    panel.border = element_rect(fill = NA, color = "black",  size = 0.5, linetype = "solid"),
    panel.background = element_rect(fill = "white", color = "white",  size = 0.5, linetype = "solid"),
    panel.grid.major = element_line(size = 0.25, linetype = 'dashed', color = "black"),
    panel.grid.minor = element_line(size = 0.1, linetype = 'dotted', color = "black")
  ) +
  xlab("Epoch") +
  ylab("Loss") +
  guides(color=guide_legend(title="Network")) +
  facet_grid(filename ~ ., scales = "free_y") +
  geom_line(size = 1) +
  geom_line(aes(x = X, y = val_acc, color=filename), size = 1, linetype = "dashed")

ggplot(data = df, aes(x = X, y = acc, color=filename)) +
  theme(
    panel.background = element_rect(fill = "white", color = "white",  size = 0.5, linetype = "solid"),
    panel.grid.major = element_line(size = 0.25, linetype = 'dashed', color = "black"),
    panel.grid.minor = element_line(size = 0.1, linetype = 'dotted', color = "black")
  ) +
  xlab("Epoch") +
  ylab("Loss") +
  guides(color=guide_legend(title="Network")) +
  facet_grid(filename ~ ., scales = "free_y") +
  geom_line(size = 1) +
  geom_line(aes(x = X, y = val_acc, color=filename), size = 1, linetype = "dashed")







# Facet by training vs loss
dfs <- lapply(data, function(df) {
  gather(df, "set", "loss", -X, -acc, -val_acc, -filename)
})
# Add filenames as a column to each dataframe
dfs <- lapply(seq_along(dfs), function(index) {
  df$filename <- rep(files[[index]], nrow(data[[index]]))
})

# Combine all of the dataframes into a master df
dfs <- dfs %>% reduce(full_join)

ggplot(data = dfs, aes(x = X, y = loss, color=set)) +
  theme(
    panel.background = element_rect(fill = "white", color = "white",  size = 0.5, linetype = "solid"),
    panel.grid.major = element_line(size = 0.25, linetype = 'dashed', color = "black"),
    panel.grid.minor = element_line(size = 0.1, linetype = 'dotted', color = "black")
  ) +
  xlab("Epoch") +
  ylab("Loss") +
  guides(color=guide_legend(title="Network")) +
  facet_grid(filename ~ ., scales = "free_y") +
  geom_line(size = 1)

