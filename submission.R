library("gbm")
library("randomForest")

train_valid_split <- function(pos_examples, neg_examples, ratio) {
	n_pos <- nrow(pos_examples)
	n_neg <- nrow(neg_examples)
	
	perm_pos_indices <- sample(1:n_pos)
	perm_neg_indices <- sample(1:n_neg)
	
	train_pos <- pos_examples[perm_pos_indices[1:floor(n_pos*ratio)],]
	train_neg <- neg_examples[perm_neg_indices[1:floor(n_neg*ratio)],]
	train_data <- rbind(train_pos, train_neg)
	
	valid_pos <- pos_examples[perm_pos_indices[-(1:floor(n_pos*ratio))],]
	valid_neg <- neg_examples[perm_neg_indices[-(1:floor(n_neg*ratio))],]
	valid_data <- rbind(valid_pos, valid_neg)	
	
	return(list(train_data, valid_data))
}

hash_row <- function(row) {
	key <- ""
	for (i in 2:191) {
		if (is.na(row[1,i])) {
			key <- paste(key, "0")
		} else {
			key <- paste(key, "1")
		}
	}
	
	return(key)
}

train_set <- read.csv("TrainingData.csv")
test_set <- read.csv("TestingData.csv")

subset_idx <- 1
remain <- train_set
file_env <- new.env()
while (nrow(remain) > 0) {
	print(paste("current remain:",nrow(remain)))
	row <- remain[1,]
	subset <- remain
	for (i in 2:191) {
		subset <- subset[is.na(subset[,i])==is.na(row[1,i]),]
	}
	
	key <- hash_row(row)
	subset_file_name <- paste(subset_idx, ".csv", sep="")
	
	subset <- subset[,colSums(is.na(subset))<nrow(subset)]
	print(paste("row: ", nrow(subset), ", col:", ncol(subset)))

	
	# sampling for unbalanced data
	pos_subset <- subset[subset[,ncol(subset)]==1,]
	neg_subset <- subset[subset[,ncol(subset)]==0,]
	
	if ((nrow(pos_subset)==0) | (nrow(neg_subset)==0)) {
		sampled_subset <- subset
	} else {
		if (nrow(neg_subset)>nrow(pos_subset)) {
			neg_subset <- neg_subset[sample(nrow(neg_subset),nrow(pos_subset)),]		
		}
		if (nrow(pos_subset)>nrow(neg_subset)) {
			pos_subset <- pos_subset[sample(nrow(pos_subset),nrow(neg_subset)),]		
		}		
		sampled_subset <- rbind(pos_subset,neg_subset)
	}
	
	sampled_subset$Labels <- as.factor(sampled_subset[,ncol(sampled_subset)])
	sampled_subset[,-c(1,ncol(sampled_subset))] <- data.matrix(sampled_subset[,-c(1,ncol(sampled_subset))])
	
	df <- data.frame(data.matrix(sampled_subset[,-c(1,ncol(sampled_subset))]))
	names(df) <- names(sampled_subset[,-c(1,ncol(sampled_subset))])
	df$Labels <- as.factor(sampled_subset[,ncol(sampled_subset)])
	
	y <- as.factor(sampled_subset[,ncol(sampled_subset)])
	X <- data.matrix(sampled_subset[,-c(1,ncol(sampled_subset))])
	
	print(paste("col: ", ncol(sampled_subset), "X col: ", ncol(X)))
	print(names(X))
	
	
	print(names(sampled_subset))
	
	if ((length(levels(y))>1) & (nrow(sampled_subset)>=50)) {
		rf <- randomForest(X, y, importance=TRUE)
		#gbm <- gbm.fit(X, y, dist="adaboost")
		#lr <- glm(Labels ~ ., family=binomial(logit), data=df)
		#print(rf)
		file_env[[key]] <- rf		
	} else {
		print(mode(subset[1,ncol(subset)]))
		file_env[[key]] <- 0
		#file_env[[key]] <- subset[1,ncol(subset)]
	}
	
	write.csv(subset, file = subset_file_name)
	print(paste("subset:", nrow(subset)))	
	remain <- remain[!(remain[,1] %in% subset[,1]),]
	subset_idx <- subset_idx+1
}

result <- data.frame(id=test_set[,1])
result$pred <- 0
for (r in 1:nrow(test_set)) {
	row <- data.matrix(test_set[r,-ncol(test_set)])
	names(row) <- names(test_set[r,-ncol(test_set)])
	key <- hash_row(row)
	row <- row[!(is.na(row))]
	row <- row[-1]
	
	if (is.null(file_env[[key]])) {
		pred <- 0
	} else {
		if (mode(file_env[[key]])=="numeric") {
			pred <- file_env[[key]]
		} else {
			#print(str(file_env[[key]]))
			#print(row[!(is.na(row))])
			#print(names(row[2:length(row)]))
			pred <- predict(file_env[[key]], row)		
		}
	}
	
	print(paste(result[r,1],": ", pred))
	print(mode(pred))
	print(class(pred))
	if (is.factor(pred)) {
		result[r,2] <- as.numeric(pred)-1
	} else {
		result[r,2] <- pred
	}
	print(paste(result[r,1],": ", result[r,2]))
}

write.csv(result, file = "submission.csv")
