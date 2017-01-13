# wikipedia-profession-classifier-part2
###Train and Evaluate Naive Bayesian Model for Profession Classification

**School:** Brandeis University  
**Course:** COSI 129a: Introduction to Big Data Analysis  
**Professors:** Marcus Verhagen, James Pustejovsky, Pengyu Hong, Liuba Shrira  
**Head TA:** Tuan Do  
**Semester:** Fall 2016  

**Team Members:** William Edgecomb, Dimokritos Stamatakis, Tyler Lichten, Nikolaos Tsikoudis 

**Description**: Our task was to train and evaluate a Naive Bayesian Model that predicts the profession associated with the text of an individual's Wikipedia article. The data for training the model was generated from part 1 of the assignment (https://github.com/wedgec/wikipedia-profession-classifier-part1) in a addition to a resource file professions.txt that couples individuals' names to their profession(s). Mahout software is used to represent feature vectors, as well as to train the model and generate predictions. For weighting features we used TF-IDF (term frequency times inverse document frequency), and we used MapReduce to calculate term frequencies and to add data to feature vectors. All processing was performed on one of Brandeis University's multi-machine clusters. For a more complete description and discussion of work for part 2, please refer to our report PDF. See also the assignment instructions. 

