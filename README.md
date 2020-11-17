# Trajectory-Data-Mining

A collection of data mining techniques of geolocation data. A canonical dataset would be the [GeoLife](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/) dataset contributed by Microsoft.

### Useful Links:
- https://www.microsoft.com/en-us/research/project/computing-with-spatial-trajectories/
- https://www.microsoft.com/en-us/research/project/trajectory-data-mining/

The book [Computing with Spatial Trajectories](https://github.com/tyqiangz/Trajectory-Data-Mining/blob/master/Useful%20Research%20Materials/Computing%20with%20Spatial%20Trajectories.pdf) written by Microsoft researchers is a treasure trove of data mining ideas for geolocation data.

I am currently looking at the paper [Mining User Similarity Based on Location History](https://github.com/tyqiangz/Trajectory-Data-Mining/blob/master/Useful%20Research%20Materials/Mining%20User%20Similarity%20Based%20on%20Location%20History.pdf) and implementing the algorithms it proposed. In their work they devised a **stay point detection algorithm (SPDA)** to detect potential stationary points where a moving object could reside in. A [notebook](https://github.com/tyqiangz/Trajectory-Data-Mining/blob/master/Stay%20Point%20Detection%20Algorithm%20Testing.ipynb) with an implementation of the algorithm explains how and when it works, a [html file](https://github.com/tyqiangz/Trajectory-Data-Mining/blob/master/Stay%20Point%20Detection%20Algorithm%20Testing.html) of the corresponding notebook is also available in case the notebook's visualisation doesn't work.

I have tried to build my own animations using matplotlib but it is too tedious to do so, especially when my main goal is to analyse the data, not so much about visualising it.

In the end, I have chose to use `folium.plugins.TimestampedGeoJson` instead.

Other interesting visualisation tools include:
- Kepler
