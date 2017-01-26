## Summary
On September 11 2001, terrorists attacked US with civilized airplanes, killing thousands of people and causing severe damages. Here I visualize the cancel rate of departure flights across US airports in 2001 and report that in September average cancel rate is much higher than any other month in US, especially in Northeast where the attacks occurred. Readers can also explore through select box to visualize monthly cancel rate at individual airport. 

## Design
I am interested in understanding the relationship between 9/11 attacks and cancel rate of departure flights in US airports. 
I realized the annual data of 2001, even after column selection, is still too huge to be processed fast enough on the browser, I decided to process the data in Python and use processed data in the visualization to accelerate it. 

In the initial design, I exported each month in a separate csv file using Python and load monthly data separately, and plot them. The visualization was not smooth since it took time to load and calculate the aggregation value. So I decided to process the data and calculated the number of flights, number of cancelled, and cancel rate before visualization for all months, and export data in a single csv, using Python. The processed file is much smaller and the visualization is much faster. I did not end up using the nest function in my final design since the data was already ready to use. 

In the preprocessing step, I also created another ‘origin’ called ‘-US’ to represent the average cancel rate by dividing total number of flights with total number of cancelled flights of all US airports. The ‘-US’ is later used in the visualization for comparison between an individual airport and national average. 

### version 1-3
To understand the performance of flights across US airports, my first design was to plot all airports on US map by combining the airport coordinates information with the flights data. I plotted airports under the alberUsa projection so that we can see all states clearly. The initial map has black stroke which was somewhat distracting. I followed the feedback given by my friend and changed the fill color to light grey and stroke color to white. 

After drawing the map, I filtered the processed annual data by month and plotted individual airports in circles.  I used the size of circles to represent the total number of flights from an airport, and used color intensity to represent the cancel rate. Initially, I used both color and size to represent the cancel rate. However, a friend suggested size for another parameter to show more relevant information. 

I used loop statement to go over all months automatically with set interval of 2000ms for readers to view the cancel rate of each month in each airport. 

### version 4-6
However, I realized that bubble chart did not well suit my goal of revealing time lapse change of cancel rate through the year. I decided to change my chart type of line chart and used dimple.js to plot cancel rate against month. I also decided to plot national average ‘-US’ first to attract audience with the title “September flights cancel rate skyrockets after 9/11 in US”, and then I created select box and allowed audience to explore different airports themselves. I used different colors to represent national average and selected airport to highlight the difference. 

### version 7-8
I calculated regional average of US airports flights delayed rates to have more insights in the geographical difference. I used python code to get the regional value and displayed regional and national value on the chart, giving audience a general introduction and comparison. Then audience can navigate with the select box to choose individual airports and compare to regional data. 

### parameter choice
In the beginning, I used delay rate rather than cancel rate and expected to see an increased delay rate in September. However, I found the opposite trend: a decreased delay rate across US in September. I think the reason for such decrease was because many flights were canceled in September, and not considered as “delay”. A more straightforward parameter is “Cancelled” column itself. 

### narrative
I created this author driven narrative to show that in year 2001, flights cancel rate skyrockets in September after 9/11 in US. The increase was higher in Northeast area where the attacks occurred, and lower in West, which was far away from the attacks. We can also observe that throughout the year, Northeast tends to have higher cancel rate in general, probably due to busy airports in the Greater New York area. Readers then can explore individual airports to compare the effect of such increase with national and regional average. 

## Feedback
### version 1-3
- Feedback person 1:
The stroke color of the map is too strong, and seems to be irrelevant information since you are not focusing on individual states. You may want to use lighter stroke colors. 

- Feedback person 2:
What does the color stand for? Does it show a certain value? Maybe adding a color bar or a legend text description helps. What does the radius stand for? Do they represent the same value or different parameters? 

- Feedback person 3:
The title showing different months in number is not as clear as text. Instead of saying “1/2001”, it may be better to say “January 2001”. 

### version 4-6
- Feedback person 4:
You display all airports all at the same time on multiple lines, but it is hard to know which is what, since there are more than 200 lines. It may be better to display one single value first.
- Feedback person 5:
I'd like to select single airport and see how it changes over month.
- Feedback person 6:
Legend? What does each color stand for? 

## Resources
- https://en.wikipedia.org/wiki/September_11_attacks
- http://stackoverflow.com/questions/17679089/pandas-dataframe-groupby-two-columns-and-get-counts
- http://stackoverflow.com/questions/12765833/counting-the-number-of-true-booleans-in-a-python-list
- http://ourairports.com/data/
- http://bl.ocks.org/mbostock/4090848
- http://www.jeromecukier.net/blog/2011/08/11/d3-scales-and-color/
- http://colorbrewer2.org/#type=diverging&scheme=RdBu&n=3
- http://eric.clst.org/Stuff/USGeoJSON
- http://bl.ocks.org/jfreels/6734823/
- http://stackoverflow.com/questions/25416063/
- https://github.com/PMSI-AlignAlytics/dimple/wiki/dimple.axis#fontSize
- http://stackoverflow.com/questions/26562129/how-to-overridemin-max-for-time-axis-in-dimple
- http://stackoverflow.com/questions/22988109/dimple-js-measure-axis-values
- http://stackoverflow.com/questions/25695917/dimple-js-or-d3-js-draw-charts-legend-at-bottom-of-svg-element
- http://jsbin.com/vecoq/8/edit?js,output
- http://www.d3noob.org/2013/01/format-date-time-axis-with-specified.html
- https://github.com/PMSI-AlignAlytics/dimple/wiki/dimple.axis
- http://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf
- http://www.infoplease.com/ipa/A0110468.html
- https://github.com/PMSI-AlignAlytics/dimple/wiki/dimple.series#addOrderRule



