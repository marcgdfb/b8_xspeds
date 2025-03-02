class PhotonCountingLegacy:
    def CheckKernelList(self, kernelList, title_kernelList="", printImages=False):
        image_binary = np.where(self.imMat > 0, 1, 0)
        rowNum, colNum = self.imMat.shape

        matrixDictionary = {}

        matConvolvedTotal = np.zeros(self.imMat.shape)
        matchCount = 0

        for kernel in kernelList:
            outputMat = np.zeros(self.imMat.shape)
            k_rows, k_cols = kernel.shape

            # Convolved the image
            for i in range(rowNum - k_rows + 1):
                for j in range(colNum - k_cols + 1):
                    # Consider areas of the same size as the kernel:
                    convolvedArea = image_binary[i:i + k_rows, j:j + k_cols]

                    # Check for an exact match of the kernel shape
                    if np.array_equal(convolvedArea, kernel):
                        # If a match is found, copy the original intensities into the output matrix
                        outputMat[i:i + k_rows, j:j + k_cols] = self.imMat[i:i + k_rows, j:j + k_cols]
                        matchCount += 1

            matConvolvedTotal += outputMat

        print(f"{matchCount} matches")

        if printImages:

            fig, ax = plt.subplots(figsize=(8, 8))
            # Get indices of nonzero elements
            y, x = np.nonzero(matConvolvedTotal)
            values = matConvolvedTotal[y, x]  # Get intensity values for color mapping

            scatter = ax.scatter(x, y, c=values, cmap='plasma', s=5, edgecolors='white', linewidth=0.2)
            plt.colorbar(scatter, ax=ax, label="Intensity")
            # Invert y-axis to match image orientation
            ax.set_ylim(ax.get_ylim()[::-1])

            # Set xticks and yticks to correspond to matrix indices
            ax.set_xticks(np.linspace(0, self.imMat.shape[1], 10))  # Set 10 evenly spaced ticks
            ax.set_yticks(np.linspace(0, self.imMat.shape[0], 10))

            ax.set_xticklabels([int(i) for i in np.linspace(0, self.imMat.shape[1], 10)])  # Ensure integer labels
            ax.set_yticklabels([int(i) for i in np.linspace(0, self.imMat.shape[0], 10)])

            ax.set_xlabel("X Index")
            ax.set_ylabel("Y Index")
            ax.set_title(title_kernelList)
            ax.set_aspect('equal')  # Ensure square pixels
            plt.show()

        return matConvolvedTotal
