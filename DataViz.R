# Load necessary libraries
library(ggplot2)
library(readr)

# Read the CSV file into a data frame
csv_file <- "/content/drive/My Drive/Output_data/intensity_results.csv"
data <- read.csv(csv_file)

output_plot <- "/content/drive/My Drive/Output_data/plot.png"  # Save plot here

# Compute IQR for outlier removal
Q1 <- quantile(data$Nucleus.Cytoplasm.Ratio, 0.25, na.rm = TRUE)
Q3 <- quantile(data$Nucleus.Cytoplasm.Ratio, 0.75, na.rm = TRUE)
IQR_value <- Q3 - Q1

# Define lower and upper bounds for outlier exclusion
lower_bound <- Q1 - 1.5 * IQR_value
upper_bound <- Q3 + 1.5 * IQR_value

# Filter data within bounds
filtered_data <- subset(data, Nucleus.Cytoplasm.Ratio >= lower_bound & Nucleus.Cytoplasm.Ratio <= upper_bound)

# Create the combined violin and boxplot
p <- ggplot(filtered_data, aes(x = Condition, y = Nucleus.Cytoplasm.Ratio, fill = Condition)) +
  geom_violin(trim = FALSE, alpha = 0.5) +
  geom_boxplot(width = 0.1, outlier.shape = NA, color = "black") +
  geom_jitter(shape = 16, position = position_jitter(0.2), size = 2, alpha = 0.7) +
  labs(title = "Violin and Boxplot of Nucleus/Cytoplasm Ratio by Condition",
       x = "Condition",
       y = "Nucleus/Cytoplasm Ratio") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "Set2")

# Now save the plot
ggsave(output_plot, plot = p, width = 8, height = 6, dpi = 400)

