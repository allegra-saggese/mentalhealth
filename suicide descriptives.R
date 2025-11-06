library(tidyr)
library(readxl)
library(dplyr)
library(ggplot2)

# load in files 
getwd()
setwd("/Users/allegrasaggese/Dropbox/Mental/Data/clean")

df1 <- if (grepl("\\.csv$", "suicide-rates-to-remove.csv")) read.csv("suicide-rates-to-remove.csv") else read_excel("file1.xlsx")
df2 <- if (grepl("\\.csv$", "2025-10-21_cafo_annual_df.csv")) read.csv("2025-10-21_cafo_annual_df.csv") else read_excel("file2.xlsx")
df3 <- read.csv("2025-08-13_mentalhealthrank_full.csv")




min(df2$FIPS_generated, na.rm = TRUE)
max(df2$FIPS_generated, na.rm = TRUE)

df1$FIPS_generated <- paste0(
  sprintf("%03d", df1$County.Code)
)

names(df1) <- tolower(names(df1))
names(df2) <- tolower(names(df2))

df1$crude.rate <- gsub("\\s+", "", df1$crude.rate)         # remove spaces
df1$crude.rate <- gsub("[^0-9.]", "", df1$crude.rate)      # keep only digits and .
df1$crude.rate <- as.numeric(df1$crude.rate)               # convert to numeric

str(df1$crude.rate)
head(df1$crude.rate)

df1$deaths_per_pop <-  df1$deaths / df1$population


df2$FIPS_generated <- sprintf("%05d", as.integer(df2$FIPS_generated))

unique(df1$year)
unique(df2$year)

df2$fips_generated <- as.character(df2$fips)


merged <- merge(df1, df2,
                by = c("fips_generated", "year"),
                all = TRUE,
                relationship = "many-to-one")

colnames(merged)
dim(merged)
colSums(!is.na(merged)) / nrow(merged) * 100

data.frame(
  column = names(merged),
  non_missing_pct = round(colSums(!is.na(merged)) / nrow(merged) * 100, 1)
) %>%
  arrange(non_missing_pct)


# test polots 
vars <- names(merged)[13:26]

for (v in vars) {
  ggplot(merged, aes_string(x = v, y = deaths)) +
    geom_point(alpha = 0.6) +
    labs(title = v, x = v, y = "Deaths") +
    theme_minimal() -> p
  print(p)
}


vars <- names(merged)[13:26]

sapply(vars, function(v)
  sum(complete.cases(merged[, c("deaths", v)]))
)


merged_complete <- merged %>% filter(!is.na(deaths), !is.na(cattle_lrg_op_inv_value_sum))
merged_complete <- merged_complete %>% filter(if_all(13:26, ~ . > 0))


vars <- names(merged_complete)[13:26]

for (v in vars) {
  df_sub <- merged_complete %>%
    filter(!is.na(.data[["deaths"]]), !is.na(.data[[v]]))
  
  if (nrow(df_sub) > 0) {
    p <- ggplot(df_sub, aes(x = .data[[v]], y = .data[["deaths"]])) +
      geom_point(alpha = 0.6) +
      labs(title = v, x = v, y = "deaths") +
      theme_minimal()
    print(p)
  } else {
    message(v, ": no overlapping non-missing data")
  }
}

for (v in vars) {
  df_sub2 <- merged_complete %>%
    filter(!is.na(.data[["crude.rate"]]), !is.na(.data[[v]]))
  
  if (nrow(df_sub2) > 0) {
    p <- ggplot(df_sub2, aes(x = .data[[v]], y = .data[["crude.rate"]])) +
      geom_point(alpha = 0.6) +
      labs(title = v, x = v, y = "deaths") +
      theme_minimal()
    print(p)
  } else {
    message(v, ": no overlapping non-missing data")
  }
}




# test plots 2 
v <- vars[2]
ggplot(merged, aes(x = .data[[v]], y = .data[["deaths"]])) +
  geom_point(alpha = 0.6) +
  labs(title = v, x = v, y = "deaths") +
  theme_minimal()




# diagnostic 

str(merged[, c(13:26, which(names(merged) == "deaths"))])
summary(merged$deaths)
colSums(!is.na(merged[13:26]))

merged <- merged %>%
  mutate(across(all_of(names(merged)[13:26]), ~as.numeric(as.character(.))),
         deaths = as.numeric(as.character(deaths)))

vars <- names(merged)[13:26]

for (v in vars) {
  p <- ggplot(merged, aes_string(x = v, y = "deaths")) +
    geom_point(alpha = 0.6) +
    labs(title = v, x = v, y = "Deaths") +
    theme_minimal()
  print(p)
}


colSums(!is.na(merged[13:26]))
summary(merged$deaths)




