# Load necessary libraries
library(ggplot2)

# Read the CSV file into a data frame
data <- read.csv("/Volumes/sils-mc/13776452/Python_scripts/Script_hoeveneers/Data_output_hoeveneers/all_intensities.csv")

# Create the combined violin and boxplot
ggplot(data, aes(x = Condition, y = Nucleus.Cytoplasm.Ratio, fill = Condition)) +
  geom_violin(trim = FALSE, alpha = 0.5) +
  geom_boxplot(width = 0.1, outlier.shape = NA, color = "black") +
  geom_jitter(shape = 16, position = position_jitter(0.2), size = 2, alpha = 0.7) +
  labs(title = "Violin and Boxplot of Nucleus/Cytoplasm Ratio by Condition",
       x = "Condition",
       y = "Nucleus/Cytoplasm Ratio") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "Set2")

