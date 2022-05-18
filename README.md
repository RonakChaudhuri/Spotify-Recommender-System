# Spotify-Recommender-System

https://docs.google.com/presentation/d/1Yz9c6PCQu5i0P4VWEw2iqdF2RP_z5l0aRUOccOZjegU/edit?usp=sharing

For my second semester Advanced Topics project, I wanted to learn about creating and coding a recommendation system, specifically for music. I wanted to expand on learning about predictive analytics and analyzing data from my last semester project. I also coded this project in Python, a language I wanted to get more familiar with, and I used the pycharm IDE. 

The first step toward going about my project was the collection of data from songs. At first, I used just a basic API I found containing just titles, artists, genres...which was pretty limiting. However I recently discovered the publicly available spotify data sets that contains many more audio features, for me to pull from, shown here such as loudness, liveness, popularity, tempo, etc. It required me to pretty much redo my entire project, however I was able to do much more with it but along with it a lot more complicated working with multiple variables 

 Not only does Spotify have these datasets available, they also have a built in python library called spotipy which gives access to all of the music data provided by the Spotify platform. To use this, I had to login to my spotify account to create an app for Spotify for Developer, giving me specific ids connected to my account. This not only allows me to query the entire spotify platform for data of different songs, albums, artists, genres, etc, but actually get ids that contain the data of my personal spotify playlists. One thing to note however is that public spotify data sets I used did not have full coverage of every song, so I had to create a function that would pull songs from the spotipy library api to match the format of the data sets. 

The next part of my project was visualizing and understand the data. This was honestly the coolest part of my project because I had a lot of fun just scraping through the data and finding very interesting trends and information about music and finding different patterns. With the Spotify datasets I used, I was able to use spotify's data of time, audio features and genres to create many different graphs that allowed me to examine trends over time and better understand the data. I used pythons Plotly library that allowed me to actually graph the information I pulled from the datasets. For example, the graph above shows the overall values of the different audio features in songs over time, and the graph below shows some genres and their overall audio features. 

Something you realize when data mining, is that there is a lot of data. There are so many songs and genres so I focused on creating a clustering algorithm that clusters the huge number of genres into a smaller amount of clusters matching them based on similarity of the audio features, which is what I used the KMeans clustering algorithm to do. However, there are still 18 different features, so it's an 18 dimensional clustering, to confine it into 2 dimensional space, I discovered the t-SNE technique or in expanded form the t-Distributed Stochastic Neighbor Embedding, and the graph on top shows the clustering of genres in two dimensions. I did the same for songs in the lower graph, however I found using the PCA technique was better as it ran much faster with larger amounts of data. 

Then it came to building the recommendation algorithm, which I specifically worked on a content based recommender system, which makes recommendations based on similarity and patterns. I first computed the average vector between the audio features of each song that is being inputted into the recommend songs function. Using that, I found the closest data points in the cluster of the average vector I found by finding the distance between vectors from a python function, and then place those points into a list to print out the results.

So to test my algorithm I placed five of some of my favorite songs, with Runaway by Kanye, Legend by Drake, Look Back at it, 90210 Hiipower, and printed out the results. Overall I was kind of disappointed as while I believed some were good matches, the accuracy was pretty rough and I realized how hard it is to actually be able to get it to be accurate as result of so much clustering and confining of data, which I did to make it easier on me to actually write the algorithm itself. So I essentially need to find a way to try my best to avoid reducing that variability while still being able to find those closest points and the similar songs. 

So overall some things I would need to work on is improving the accuracy of the results, as well as the run time because it takes quite some time for my algorithm to run. Another part I would like to work on is another type of recommendation system which is the collaborative filtering system that actually recommends based on finding users similar to you to recommend songs based on those similar users. Lastly, I am currently working on doing this recommendation system through actual users spotify playlists. Overall, I had a lot of fun with this project and am satisfied about learning a lot from this topic and what I was able to accomplish.

