# Calibrate(imTest,None).computeLine(7.063102423636962e-05,862,cBounds=cBoundsL2,plotGraph=True,cPlotVal=1418,plotResults=False)

# Calibrate(imTest).fitGaussianToLineIntegral(0.00005,1000,cBounds=cBoundsL2,plotGauss=True)

# Calibrate(imTest,line_right_txt).optimiseLines(aBounds=aBoundsL2,bBounds=bBoundsL2,cBounds=cBoundsL2, plotGraph=True, plotResults=True)
# Calibrate(imTest, line_left_txt).optimiseLines(aBounds=aBoundsL1, bBounds=bBoundsL1, cBounds=cBoundsL1,
#                                                 plotGraph=True, plotResults=True)

cBoundsL1 = (1200, 1360)
aBoundsL1 = (0.00001, 0.0001)
# The Line ends further along x at the bottom than it does y
bBoundsL1 = (700, 1024)

cBoundsL2 = (1380, 1460)
aBoundsL2 = (0.00001, 0.0001)
# The Line ends further along x at the bottom than it does y
bBoundsL2 = (700, 1024)