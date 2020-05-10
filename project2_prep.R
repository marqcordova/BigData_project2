#Just run this script once
library(tidyverse)
library(here)

list.files(here())

train <- read_csv('train_labels.csv') %>% 
  mutate(filename = paste(id,'.tif',sep=''))

#Move all of the orginal training images to a folder called "kaggle_train"
#we won't need them, but I don't want to purge them
if ("0446e9309842878431e6968b7063738af7e7dc78.tif" %in% list.files(here('test'))){
  dir.create("kaggle_test", showWarnings = FALSE)
  kaggle_test_ims <- list.files(here('test'), pattern = '.tif')
  file.copy(from = here('test', kaggle_test_ims),
            to   = here('kaggle_test', kaggle_test_ims))
  file.remove(here('test', kaggle_test_ims))
} else {
  print('training images already moved')
}

########################################################
#Make train, val, test sets from the labelled training images
########################################################
set.seed(1234)
val <- train %>% 
  sample_n(40000)
write_csv(val, 'val.csv')

test <- train %>% 
  filter(!id %in% val$id) %>% 
  sample_n(40000)
write_csv(test, 'test.csv')
  
train2 <- train %>% 
  filter(!id %in% val$id,
         !id %in% test$id)
write_csv(train2, 'train.csv')

########################################################
#Make train, val, test folders and move the images there
########################################################
dir.create("project_train", showWarnings = FALSE)
dir.create("project_test", showWarnings = FALSE)
dir.create("project_val", showWarnings = FALSE)

file.copy(from = here('train', train2$filename),
          to   = here('project_train', train2$filename))

file.copy(from = here('train', val$filename),
          to   = here('project_val', val$filename))

file.copy(from = here('train', test$filename),
          to   = here('project_test', test$filename))