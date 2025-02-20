# Load necessary libraries
library(ggplot2)
library(readr)

# Read the CSV file into a data frame
csv_file <- "/content/drive/My Drive/Output_data/intensity_results.csv"
data <- read.csv(csv_file)

output_plot <- "/content/drive/My Drive/Output_data/plot.png"  # Save plot here

# Create the combined violin and boxplot
p <- ggplot(data, aes(x = Condition, y = Nucleus.Cytoplasm.Ratio, fill = Condition)) +
  geom_violin(trim = FALSE, alpha = 0.5) +
  geom_boxplot(width = 0.1, outlier.shape = NA, color = "black") +
  geom_jitter(shape = 16, position = position_jitter(0.2), size = 2, alpha = 0.7) +
  labs(title = "Violin and Boxplot of Nucleus/Cytoplasm Ratio by Condition",
       x = "Condition",
       y = "Nucleus/Cytoplasm Ratio") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "Set2")

# Force the plot to fully render before saving
print(p)  # Print the plot first
dev.off()  # Close any open graphic device

# Now save the plot
ggsave(output_plot, plot = p, width = 10, height = 6, dpi = 300)

