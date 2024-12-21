library(tidyverse)
library(ggmap)

predictors

seattle_bbox <- c(left = -122.459, bottom = 47.481, right = -122.224, top = 47.735)

seattle_map <- get_stadiamap(bbox = seattle_bbox, zoom = 12, maptype = "stamen_terrain")

register_stadiamaps("88b6d7f3-77d4-4be7-b620-4aaabad3cc71", write = FALSE)

ggmap(seattle_map) +
  geom_point(data = airbnb_df, aes(x = longitude, y = latitude, color = price), size = 2) +
  scale_color_viridis_c(option = "magma", name = "Price") +
  labs(title = "Seattle Airbnb Homestay Prices",
       x = "Longitude",
       y = "Latitude") +
  theme_minimal()
