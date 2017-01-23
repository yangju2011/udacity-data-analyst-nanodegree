## Summary
On September 11 2001, terrorists attacked US with civilized airplanes, killing thousands of people and causing severe damages. Here I visualize the cancel rate of departure flights across US in 2001 and report that in September cancel rate is much higher than any other months across US, especially in northeast region near New York. 9/11 attacks largely increases flight cancel rate. 

## Design
I am interested in understanding the relationship between 9/11 attacks and cancel rate of departure flights in US airports. 
I realized the annual data of 2001, even after column selection, is still too huge to be processed fast enough on the browser, I decided to process the data in Python and use processed data in the visualization to accelerate it. 

In the initial design, I exported each month in a separate csv file using Python and load monthly data separately, and plot them. The visualization was not smooth since it took time to load and calculate the aggregation value. So I decided to process the data and calculated the number of flights, number of cancelled, and cancel rate before visualization for all months, and export data in a single csv, using Python. The processed file is much smaller and the visualization is much faster. I did not end up using the nest function in my final design since the data was already ready to use. 

To understand the performance of flights across US airports, I first plotted all airports on US map by combining the airport coordinates information with the flights data. I plotted airports under the alberUsa projection so that we can see all states clearly. The initial map has black stroke which was somewhat distracting. I followed the feedback given by my friend and changed the fill color to light grey and stroke color to white. 

After drawing the map, I filtered the processed annual data by month and plotted individual airports in circles.  I used the size of circles to represent the total number of flights from an airport, and used color intensity to represent the cancel rate. I used single tone color 'red' with its intensity corresonding to cancel rate. Initially, I used both color and size to represent the cancel rate. However, a friend suggested size for another parameter to show more relevant information. 

I used loop statement to go over all months automatically with set interval of 2000ms for readers to view the cancel rate of each month in each airport. 

In the beginning, I used delay rate rather than cancel rate and expected to see an increased delay rate in September. However, I found the opposite trend: a decreased delay rate across US in September. I think the reason for such decrease was because many flights were canceled in September, and not considered as “delay”. A more straightforward parameter is “Cancelled” column itself. 

I created this author driven narrative to show that through the year 2001, cancel rate was in general low except in September, when many airports experienced high cancel rate, and in northeast such as New York, the cancel rate was the highest. 

## Feedback
- Feedback person 1:
The stroke color of the map is too strong, and seems to be irrelevant information since you are not focusing on individual states. You may want to use lighter stroke colors. 

- Feedback person 2:
What does the color stand for? Does it show a certain value? Maybe adding a color bar or a legend text description helps. What does the radius stand for? Do they represent the same value or different parameters? 

- Feedback person 3:
The title showing different months in number is not as clear as text. Instead of saying “1/2001”, it may be better to say “January 2001”. 

## Resources
- https://en.wikipedia.org/wiki/September_11_attacks
- http://stackoverflow.com/questions/17679089/pandas-dataframe-groupby-two-columns-and-get-counts
- http://stackoverflow.com/questions/12765833/counting-the-number-of-true-booleans-in-a-python-list
- http://ourairports.com/data/
- http://bl.ocks.org/mbostock/4090848
- http://www.jeromecukier.net/blog/2011/08/11/d3-scales-and-color/
- http://colorbrewer2.org/#type=diverging&scheme=RdBu&n=3
- http://eric.clst.org/Stuff/USGeoJSON

