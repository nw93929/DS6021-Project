if (!requireNamespace("baseballr", quietly = TRUE)) {
  install.packages("baseballr")
}
library(baseballr)
library(dplyr)
library(lubridate)
library(rstudioapi)
library(readr)

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

combined_free_agents <- combined_free_agents %>%
  select(season, everything())  # puts 'season' first

# set working directory and create file path
setwd(dirname(getActiveDocumentContext()$path))
path <- file.path(getwd(), "data", "raw_data", "MLB_stats_free_agents.csv")

# create output file
write_csv(combined_free_agents, path)

