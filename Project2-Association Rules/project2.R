library(arules)
library(data.table)
library(readr)
library(arulesViz)
library(RColorBrewer)
# import data
trnsact <- read_csv("trnsact.csv", col_names = FALSE, 
                           col_types = cols(X10 = col_skip(), X11 = col_skip(), 
                                                    X12 = col_skip(), X13 = col_skip(), 
                                                    X14 = col_skip(), X3 = col_skip(), 
                                                    X5 = col_skip(), X8 = col_skip(), 
                                                    X9 = col_skip()))
# change the colum names
names(trnsact) <- c("SKU","store","trannum","date","stype")

#randomly choose data record
transaction <- trnsact[sample(row.names(trnsact), 5000000),]

#modify the data
transaction <- subset(transaction,stype == "P" )
transaction <- transaction[,-c(5)]
transaction$SKU <- as.factor(transaction$SKU)
transaction$store <- as.numeric(transaction$store)
transaction$trannum <- as.numeric(transaction$trannum)
transaction$date <- as.Date(transaction$date)
transaction$basket<- paste(transaction$store,transaction$trannum,transaction$date,collapse = NULL, sep = ',')
transaction$basket<- as.factor(transaction$basket)
transaction1<- data.frame(transaction$basket,transaction$SKU)

# output the transaction data and then import as transaction form
write.csv(transaction1,file="D:/data science/Project2-Association Rules/project 2/transaction1.csv",row.names = F)
transaction2 <- read.transactions("./transaction1.csv",cols=c(1,2), format="single",rm.duplicates=TRUE, sep=',') 
summary(transaction2)

# head(itemFreq[order(-itemFreq)],10)
itemFrequencyPlot(transaction2, topN=10, horiz=T)

# association rules
rules=apriori(transaction2,parameter=list(support=0.000008,confidence=0.01))

#exclude redundant rules 
subset.matrix<-is.subset(rules,rules,sparse = FALSE)
subset.matrix[lower.tri(subset.matrix,diag = T)]<-NA
redundant<-colSums(subset.matrix,na.rm=T)>=1
which(redundant)
rules.pruned<- rules[!redundant]
ordered_rules <- sort(rules.pruned, by="lift")
inspect(ordered_rules)

#rules visualization 
plot(ordered_rules[1:20],
     control=list(jitter=2,col=rev(brewer.pal(9, "Greens")[4:9])),
     shading = 'lift')
