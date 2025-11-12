library(tidyr)
library(readxl)
library(dplyr)
library(ggplot2)

# load in files 
getwd()
setwd("/Users/allegrasaggese/Dropbox/Mental/Data/clean")

df1 <- if (grepl("\\.csv$", "suicide-rates-to-remove.csv")) read.csv("suicide-rates-to-remove.csv") else read_excel("file1.xlsx")
df2 <- if (grepl("\\.csv$", "2025-11-12_cafo_annual_df.csv")) read.csv("2025-11-12_cafo_annual_df.csv") else read_excel("file2.xlsx")
df3 <- read.csv("2025-10-17_mentalhealthrank_full.csv")

# standardize cols and fips code 
names(df1) <- tolower(names(df1))
names(df2) <- tolower(names(df2))

# convert county.code to string and pad to 5 digits
df1$county.code <- sprintf("%05s", as.character(df1$county.code))
df2$fips_generated <- sprintf("%05s", as.character(df2$fips_generated))

class(df1$year)
class(df2$year)

names(df1)[names(df1) == "county.code"] <- "fips_generated"
df1 <- df1 %>%
  filter(fips_generated != "000NA")

summary(df2)
sapply(df2, function(x) sum(is.na(x)) / length(x) * 100)
sapply(df2, function(x) length(unique(x)))
df2 %>%
  select(where(is.character)) %>%
  summarise(across(everything(), ~sum(grepl("^\\s|\\s$", .x, perl = TRUE))))
sapply(df2, function(x) any(grepl("[A-Za-z]", x) & grepl("\\d", x)))
sapply(df2, function(x) if(is.character(x)) sum(x == "", na.rm=TRUE) else NA) # problem w coallesce is that there is non-NA empty strings that are being considered strings 

df2 <- df2 %>%
  mutate(across(where(is.character), ~na_if(trimws(.x), "")))

# coallesce the size cols into one classification
size_cols <- grep("size_class$", names(df2), value = TRUE)
problem_rows <- df2 %>%
  filter(!is.na(animal_type) & if_all(all_of(size_cols), is.na)) # empty - working now 

df2 <- df2 %>%
  mutate(size = coalesce(!!!syms(size_cols))) %>%
  select(-all_of(size_cols))

# merge on fips_generated and year
merged_df <- merge(df2, df1, by = c("fips_generated", "year"), all = TRUE)
print(unique(merged_df$size))

n_total  <- nrow(merged_df)
n_large  <- sum(merged_df$size == "large", na.rm = TRUE)
n_medium <- sum(merged_df$size == "medium", na.rm = TRUE)
n_small  <- sum(merged_df$size == "small", na.rm = TRUE)
n_missing <- sum(is.na(merged_df$size))

pct_large  <- n_large / n_total * 100
pct_medium <- n_medium / n_total * 100
pct_small  <- n_small / n_total * 100
pct_missing <- n_missing / n_total * 100

data.frame(
  n_total, n_large, n_medium, n_small, n_missing,
  pct_large = round(pct_large, 2),
  pct_medium = round(pct_medium, 2),
  pct_small = round(pct_small, 2),
  pct_missing = round(pct_missing, 2)
)
##### MAKE SUICIDE PLOTS --- start with crude rate vs. hog CAFOS
year_target <- 2012

# list all animal types so that I can create a graph 
unique(merged_df$animal_type)
animal_types <- unique(trimws(tolower(merged_df$animal_type)))
print(animal_types)

df_year_TEST <- merged_df %>% filter(year == year_target)

for (a in animal_types) {
  df_sub <- df_year_TEST %>% filter(animal_type == a)
  
  p <- ggplot(df_sub, aes(x = count_cafo, y = crude_rate, color = size)) +
    geom_point(alpha = 0.6) +
    scale_color_manual(values = c("small" = "gray60", "medium" = "steelblue", "large" = "darkred")) +
    labs(
      title = paste("CAFO ops vs. suicide rates (", a, ")", sep = ""),
      x = "Count of CAFO operations",
      y = "Crude rate (per 100k)",
      color = "Size Category"
    ) +
    theme_minimal()
  
  print(p)
}

##### LOOP OVER A SET OF YEARS 
years_to_plot <- c(2005, 2008, 2011, 2016)

for (a in animal_types) {
  df_sub <- merged_df %>%
    filter(animal_type == a, year %in% years_to_plot)
  
  if (nrow(df_sub) == 0) {
    message("No data for ", a)
    next
  }
  
  p <- ggplot(df_sub, aes(x = count_cafo, y = crude_rate, color = size)) +
    geom_point(alpha = 0.6) +
    facet_wrap(~ year, scales = "free") +
    scale_color_manual(values = c("small" = "gray60",
                                  "medium" = "steelblue",
                                  "large" = "darkred")) +
    labs(
      title = paste("CAFO vs Suicide Rates —", a),
      x = "Count of CAFO operations",
      y = "Crude suicide rate (per 100k)",
      color = "Size"
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      strip.text = element_text(size = 9),
      plot.title = element_text(size = 11)
    )
  
  print(p)
}


# SIZE plots 
# assign broader category 
unique(merged_df$animal_type)
merged_df <- merged_df %>%
  mutate(
    animal_group = case_when(
      animal_type %in% c("cattle_ops_size_inv_size_class", "cattle_head_size_inv_size_class",
                         "dairy_head_size_inv_size_class", "cattle_500lbs_ops_size_sales_size_class", 
                         "cattle_calves_ops_size_sales_size_class",  "beef_map_head_size_inv_size_class", 
                         "cattle_feed_ops_size_inv_size_class",  "dairy_ops_size_inv_size_class",
                         "cattle_senzcow_ops_size_inv_size_class", "beef_ops_size_inv_size_class", 
                         "calves_ops_size_sales_size_class",  "cattle_feed_ops_size_sales_size_class", 
                         "cattle_feed_map_head_size_inv_size_class") ~ "cattle",
      animal_type %in% c("hog_head_size_inv_size_class", "breed_hog_ops_size_inv_size_class" , 
                         "hog_ops_size_inv_size_class", "hog_ops_size_sales_size_class") ~ "swine",
      animal_type %in% c("layer_ops_size_size_class", "broiler_head_size_size_class" ) ~ "chicken",
      TRUE ~ "other"
    )
  )


sizes <- c("small", "medium", "large")

# loop over size levels
for (s in sizes) {
  df_sub <- merged_df %>% 
    filter(size == s) 
  
  if (nrow(df_sub) == 0) {
    message("No data for size category: ", s)
    next
  }
  
  p <- ggplot(df_sub, aes(x = count_cafo, y = crude_rate, color = animal_group)) +
    geom_point(alpha = 0.6) +
    scale_color_manual(values = c("poultry" = "goldenrod3",
                                  "swine" = "firebrick",
                                  "cattle" = "steelblue",
                                  "other" = "gray60")) +
    labs(
      title = paste("CAFO Count vs Suicide Rates — Size:", s),
      x = "Count of CAFO operations",
      y = "Crude suicide rate (per 100k)",
      color = "Animal Group"
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      plot.title = element_text(size = 11)
    )
  
  print(p)
}



