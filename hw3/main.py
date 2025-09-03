import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


class Evaluator:
    """
    A class for evaluating a biometric system's performance.
    """
    def __init__(self, epsilon, num_thresholds, genuine_scores, impostor_scores, plot_title):
        self.epsilon = epsilon
        self.num_thresholds = num_thresholds
        self.genuine_scores = genuine_scores
        self.impostor_scores = impostor_scores
        self.plot_title = plot_title
        self.thresholds = np.linspace(-0.1, 1.1, num_thresholds)

    def get_dprime(self):
        """
        Calculate the d' (d-prime) metric.

        Returns:
        - float: The calculated d' value.
        """
        mean_genuine = np.mean(self.genuine_scores)
        mean_impostor = np.mean(self.impostor_scores)
        std_genuine = np.std(self.genuine_scores)
        std_impostor = np.std(self.impostor_scores)

        x = abs(mean_genuine - mean_impostor)
        y = np.sqrt((std_genuine**2 + std_impostor**2) / 2)
        return x / (y + self.epsilon)

    def plot_score_distribution(self):
        """
        Plot the distribution of genuine and impostor scores.
        """
        plt.figure()
        
        # Plot the histogram for genuine scores
        plt.hist(
            # Provide genuine scores data here
            # color: Set the color for genuine scores
            # lw: Set the line width for the histogram
            # histtype: Choose 'step' for a step histogram
            # hatch: Choose a pattern for filling the histogram bars
            # label: Provide a label for genuine scores in the legend
            
            self.genuine_scores, color='green', lw=2, histtype='step', hatch='/', label='Genuine'
        )
        
        # Plot the histogram for impostor scores
        plt.hist(
            # Provide impostor scores data here
            # color: Set the color for impostor scores
            # lw: Set the line width for the histogram
            # histtype: Choose 'step' for a step histogram
            # hatch: Choose a pattern for filling the histogram bars
            # label: Provide a label for impostor scores in the legend

            self.impostor_scores, color='red', lw=2, histtype='step', hatch='\\', label='Impostor'
        )
        
        # Set the x-axis limit to ensure the histogram fits within the correct range
        plt.xlim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        
        # Add legend to the upper left corner with a specified font size
        plt.legend(
            # loc: Specify the location for the legend (e.g., 'upper left')
            # fontsize: Set the font size for the legend

            loc='upper left', fontsize=10
        )
        
        # Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            # Provide the x-axis label
            # fontsize: Set the font size for the x-axis label
            # weight: Set the font weight for the x-axis label

            'Score', fontsize=12, weight='bold'
        )
        
        plt.ylabel(
            # Provide the y-axis label
            # fontsize: Set the font size for the y-axis label
            # weight: Set the font weight for the y-axis label

            'Frequency', fontsize=12, weight='bold'
        )
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set font size for x and y-axis ticks
        plt.xticks(
            # fontsize: Set the font size for x-axis ticks

            fontsize=10
        )
        
        plt.yticks(
            # fontsize: Set the font size for y-axis ticks

            fontsize=10
        )
        
        # Add a title to the plot with d-prime value and system title
        plt.title('Score Distribution Plot\nd-prime= %.2f\nSystem %s' % 
                  (self.get_dprime(), 
                   self.plot_title),
                  fontsize=15,
                  weight='bold')
        
       
        # Save the figure before displaying it
        plt.savefig('score_distribution_plot_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        
        # Display the plot after saving
        plt.show()
        
        # Close the figure to free up resources
        plt.close()

        return

    def get_EER(self, FPR, FNR):
        """
        Calculate the Equal Error Rate (EER).
    
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - FNR (list or array-like): False Negative Rate values.
    
        Returns:
        - float: Equal Error Rate (EER).
        """
        EER = 0
        
        # Add code here to compute the EER
        for i in range(len(FPR)):
            if FPR[i] == FNR[i]:
                EER = FPR[i]
                break
            elif FPR[i] < FNR[i]:
                EER = FPR[i]
                break
            else:
                EER = FNR[i]
        
        return EER

    def plot_det_curve(self, FPR, FNR):
        """
        Plot the Detection Error Tradeoff (DET) curve.
        Parameters:
         - FPR (list or array-like): False Positive Rate values.
         - FNR (list or array-like): False Negative Rate values.
        """
        
        # Calculate the Equal Error Rate (EER) using the get_EER method
        EER = self.get_EER(FPR, FNR)
        
        # Create a new figure for plotting
        plt.figure()
        
        # Plot the Detection Error Tradeoff Curve
        plt.plot(
            # FPR values on the x-axis
            # FNR values on the y-axis
            # lw: Set the line width for the curve
            # color: Set the color for the curve
            FPR, FNR, lw=2, color='blue', label="Detection Error Tradeoff (DET) curve"
        )
        
        # Add a text annotation for the EER point on the curve
        plt.text(EER + 0.07, EER + 0.07, "EER", style='italic', fontsize=12,
                 bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        # Plot the diagonal line representing random classification
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        # Scatter plot to highlight the EER point on the curve
        plt.scatter([EER], [EER], c="black", s=100)

        plt.text(EER + 0.07, EER + 0.07, "EER", style='italic', fontsize=12,
                 bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        plt.scatter([EER], [EER], c="black", s=100)
        
        # Set the x and y-axis limits to ensure the plot fits within the range 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(
            # color: Set the color of grid lines
            # linestyle: Choose the line style for grid lines
            # linewidth: Set the width of grid lines
            color='gray', linestyle='--', linewidth=0.5
        )
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            # 'False Pos. Rate': Set the x-axis label
            # fontsize: Set the font size for the x-axis label
            # weight: Set the font weight for the x-axis label
            'False Positive Rate', fontsize=12, weight='bold'
        )
        
        plt.ylabel(
            # 'False Neg. Rate': Set the y-axis label
            # fontsize: Set the font size for the y-axis label
            # weight: Set the font weight for the y-axis label
            'False Negative Rate', fontsize=12, weight='bold'
        )
        
        # Add a title to the plot with EER value and system title
        plt.title(
            # 'Detection Error Tradeoff Curve \nEER = %.5f\nSystem %s': Set the title
            # EER: Provide the calculated EER value
            # self.plot_title: Provide the system title
            # fontsize: Set the font size for the title
            # weight: Set the font weight for the title
            'Detection Error Tradeoff Curve \nEER = %.5f\nSystem %s' % (EER, self.plot_title), fontsize=15, weight='bold'
        )
        
        # Set font size for x and y-axis ticks
        plt.xticks(
            # fontsize: Set the font size for x-axis ticks

            fontsize=10
        )
        
        plt.yticks(
            # fontsize: Set the font size for y-axis ticks

            fontsize=10
        )
        
        # Save the plot as an image file
        plt.savefig(
            # 'det_curve_%s.png': Set the filename for saving the plot
            # self.plot_title: Provide the system title for the filename
            'det_curve_%s.png' % self.plot_title, dpi=300, bbox_inches="tight"
        )
        
        # Display the plot
        plt.show()      
        
        # Close the plot to free up resources
        plt.close()
    
        return

    def plot_roc_curve(self, FPR, TPR):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - TPR (list or array-like): True Positive Rate values.
        """
        auc_roc = metrics.auc(FPR, TPR)
        # Create a new figure for the ROC curve
        plt.figure()
        # Plot the ROC curve using FPR and TPR with specified attributes
        plt.plot(FPR, TPR, lw=2, color='blue', label="ROC Curve")
        # Set x and y axis limits, add grid, and remove top and right spines
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        # Set labels for x and y axes, and add a title
        plt.xlim([-0.05, 1.05])
        # Set font sizes for ticks, x and y labels
        plt.ylim([-0.05, 1.05])
        # Save the plot as a PNG file and display it
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        # Close the figure to free up resources
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel('False Positive Rate', fontsize=12, weight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, weight='bold')
        plt.title(f'Receiver Operating Characteristic Curve\nArea under Curve = {auc_roc:.5f}\nSystem {self.plot_title}', fontsize=15, weight='bold')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig('roc_curve_%s.png' % self.plot_title, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        return

    def compute_rates(self):
        """
        Compute FPR, FNR, and TPR for the score thresholds.
        """
        FPR = []
        FNR = []
        TPR = []
        for threshold in self.thresholds:
            TP = np.sum(self.genuine_scores >= threshold)
            FN = np.sum(self.genuine_scores < threshold)
            FP = np.sum(self.impostor_scores >= threshold)
            TN = np.sum(self.impostor_scores < threshold)

            FPR.append(FP / (FP + TN + self.epsilon))
            FNR.append(FN / (TP + FN + self.epsilon))
            TPR.append(TP / (TP + FN + self.epsilon))
        return FPR, FNR, TPR

def main():

    # Set the random seed to 1.
    np.random.seed(1)

    # Name the systems A, B, and C
    systems = ["A", "B", "C"]

    for system in systems:
        
        # Use np.random.random sample() to generate a random float between 
        # 0.5 and 0.9 and another random float between 0.0 and 0.2. Use these 
        # as the μ (mean) and σ (standard deviation), respectively, to generate 
        # 400 genuine scores using np.random.normal()
        low = 0.5
        high = 0.9
        genuine_mean = low + (high - low) * np.random.sample()
        low = 0.0
        high = 0.2
        genuine_std = low + (high - low) * np.random.sample() 
        genuine_scores = np.random.normal(genuine_mean, genuine_std, 400) 
        
        # Repeat with μ ∈ [0.1, 0.5) and σ ∈ [0.0, 0.2) to generate 1,600 
        # impostor scores
        low = 0.1
        high = 0.5
        impostor_mean = low + (high - low) * np.random.sample()
        low = 0.0
        high = 0.2
        impostor_std = low + (high - low) * np.random.sample()
        impostor_scores = np.random.normal(impostor_mean, impostor_std, 1600)
        
        # Creating an instance of the Evaluator class
        evaluator = Evaluator(
            epsilon=1e-12,
            num_thresholds=200,
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
            plot_title="%s" % system
        )
        
        # Generate the FPR, FNR, and TPR using 200 threshold values equally spaced
        # between -0.1 and 1.1.
        FPR, FNR, TPR = evaluator.compute_rates()
    
        # Plot the score distribution. Include the d-prime value in the plot’s 
        # title. Your genuine scores should be green, and your impostor scores 
        # should be red. Set the x axis limits from -0.05 to 1.05
        evaluator.plot_score_distribution()
                
        # Plot the DET curve and include the EER in the plot’s title. 
        # Set the x and y axes limits from -0.05 to 1.05.
        evaluator.plot_det_curve(FPR, FNR)
        
        # Plot the ROC curve. Set the x and y axes limits from -0.05 to 1.05.
        evaluator.plot_roc_curve(FPR, TPR)
        
        
if __name__ == "__main__":
    main()

