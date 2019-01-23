library(plyr)
library(cluster)
library(ggplot2)
library(gcookbook)
library(caret)
library(MASS)
library(HSAUR2)
library(factoextra)
library(sjPlot)
library(fpc)
# import the data
data2 <-read.csv(file.choose())

# rename columns we need to simplify
names(data2)[1] <- "NPI"
names(data2)[12] <- "state"
names(data2)[20] <- "service"
names(data2)[23] <- "allow"
names(data2)[24] <- "submit"
names(data2)[26] <- "standardized"

# aggregat features by states
state.service <-aggregate(service ~ state, data = data2, sum)
state.allow <-aggregate(allow ~ state, data = data2, sum)
state.submit <-aggregate(submit ~ state, data = data2, sum)
state.standardized <-aggregate(standardized ~ state, data = data2, sum)

# calculate the total allow, submit and standardized amount.
average_allow <-state.allow["allow"] / state.service["service"]
average_submit <-state.submit["submit"] / state.service["service"]
average_standardized <-state.standardized["standardized"] / state.service["service"]

#delete duplication of NPI
NPI =data2[!duplicated(data2[,c(1)]),]
state.NPI = count(NPI, "state")

#merge
state <-data.frame (state.NPI, state.service["service"],
                    average_allow, average_submit,
                    average_standardized)

# calculate the Medicare fee cover rate
state$Medicare_fee_cover_rate <- state.standardized$standardized / state.submit$submit

#delete states not in U.S.A
state <-state[-c(1,2,5,7,16,31,46,54,60,61),]
names(state)[2] <- "provider"

# histogram
par(mfrow=c(2,3))
ggplot(state,  aes(x=provider)) + 
  geom_histogram(stat="bin", fill="lightblue", colour="black" ) + 
  labs(x="Number of Providers", y="Frequency", title="Frequency of Providers in States")
ggplot(state,  aes(x=service)) + 
  geom_histogram(stat="bin", fill="lightblue", colour="black" ) + 
  labs(x="Number of Service", y="Frequency", title="Frequency of Service in States") 
ggplot(state,  aes(x=allow)) + 
  geom_histogram(stat="bin", fill="lightblue", colour="black" ) + 
  labs(x="Average Amount of Allow", y="Frequency", title="Frequency of Average Amount of Allow in States")
ggplot(state,  aes(x=submit)) + 
  geom_histogram(stat="bin", fill="lightblue", colour="black" ) + 
  labs(x="Average Amount of Submit", y="Frequency", title="Frequency of Average Amount of Submit in States")
ggplot(state,  aes(x=standardized)) + 
  geom_histogram(stat="bin", fill="lightblue", colour="black" ) + 
  labs(x="Average Amount of Standardized", y="Frequency", title="Frequency of Average Amount of Standardized in States")

# exclude outliers
rownames(state)<-NULL
state <-state[-c(5,10,35,44),]
rownames(state)<-NULL
structure(state)
# data preprocessing
# normlization
state$allow <- state$allow * state$service
state_scale <-scale(state[,-c(1)])

# SSE find the best K
sjc.elbow(state_scale[,c(1:3)], steps = 15, show.diff = FALSE)

# Gap Statistic find K
clusGap(state_scale[,c(1:3)], kmeans, 10, B = 47, verbose = interactive())

# K means clustering
k_mean = kmeans(state_scale[,c(1:3)], 4)
state_scale <- as.data.frame(state_scale)
k_mean$cluster <- factor(k_mean$cluster)
ggplot() + geom_point(data = state_scale,aes(x = Medicare_fee_cover_rate,y = provider,col=k_mean$cluster),size=6)+
  geom_point(data = state_scale,aes(x = Medicare_fee_cover_rate,y = service,colour=k_mean$cluster),size=6)+
  geom_point(data = state_scale,aes(x = Medicare_fee_cover_rate,y = allow,col=k_mean$cluster),size=6) +
  labs( y="Range")
k_mean$cluster
k_mean$centers
k_mean$size

# asses the quality of my clustering
dis <- dist(state_scale[,c(1:3)])
sil <- silhouette (k_mean$cluster, dis)
summary(sil)
plot(sil)


