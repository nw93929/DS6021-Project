if (!requireNamespace("baseballr", quietly = TRUE)) {
  install.packages("baseballr")
}
library(baseballr)
library(dplyr)
library(lubridate)

# Vector of years
years <- 2003:2017

# Initialize list to store yearly tibbles
all_free_agents <- list()

# Loop over years
for (yr in years) {
  # Fetch free-agent data for that year
  fa <- mlb_people_free_agents(season = yr)
  
  # Convert to tibble and add columns
  fa_tbl <- as_tibble(fa) %>%
    mutate(
      date_signed = as.Date(date_signed),  # ensure Date type
      signed_year = year(date_signed),     # year extracted from date_signed
      season = yr                          # add loop year as 'season' column
    )
  
  # Add to list without filtering
  all_free_agents[[as.character(yr)]] <- fa_tbl
}

# Combine all years into one tibble
combined_free_agents <- bind_rows(all_free_agents)

# Using base R
write.csv(combined_free_agents, "combined_free_agents_2014_2024.csv", row.names = FALSE)

# Or using readr for faster writing
# install.packages("readr")  # if not installed
combined_free_agents <- combined_free_agents %>%
  select(season, everything())  # puts 'season' first, keeps all other columns after


setwd("C:/Users/garre/school/ds_6021/project/DS6021-Project")
library(readr)
write_csv(combined_free_agents, "combined_free_agents_2014_2024.csv")

