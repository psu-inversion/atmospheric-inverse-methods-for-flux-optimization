
library(geoR)
library(MASS)

NX <- 40
NY <- 30
NT <- 1
NRUNS <- 100

display.plots = FALSE

per.run <- NT * NY * NX

#' X-axis limits for variogram plots
LIMITS.X = c(0, 40)
#' Y-axis limits for variogram plots (scaled)
LIMITS.Y = c(0, 2)

ident.increments <- as.data.frame(
    scan("identical_gaussian_5.csv", skip=1, sep=",",
         what=list(iteration=5, obs.time="", y=1., x=1., increment=.3))[c(1, 3, 4, 5)])

ident.fitted.vals <- matrix(nrow=NRUNS, ncol=2)

ident.variog.file = "identical_gaussian_5_variogs.pdf"
if (! display.plots) {
    pdf(file=ident.variog.file)
}

for (i in 1:NRUNS) {
    run.end <- i * per.run
    run.start <- run.end - per.run + 1
    incr <- as.geodata(as.data.frame(ident.increments[run.start:run.end, 1:4]),
                       coords.col=c(2, 3), data.col=4, realisations=1)

    incr.var <- var(incr$data)
    # Get the variogram
    var2 <- variog(incr)
    plot(var2, scaled=TRUE, xlim=LIMITS.X, ylim=LIMITS.Y,
         bty="n", xaxt="n", yaxt="n", xlab="", ylab="")
    par(new=TRUE)
    # Fit a model to the variogram
    # Gaussian covariances may produce fields too smooth for a real process
    # and also have convergence problems. Gneiting covariances are visiually 
    # similar and do not have that problem
    fit.variog.ident <- variofit(var2, cov.model="gneiting",
                                 ini.cov.pars=data.frame(sigmasq=incr.var,
                                                         phi=5),
                                 control=list(parscale=c(incr.var, 1, 1)))
    # Use the fitted models to start a restricted maximum likelihood
    # search using the original data
    fit.data.ident <- likfit(incr, ini.cov.pars=fit.variog.ident,
                             cov.model="gneiting", lik.method="REML",
                             control=list(parscale=c(incr.var, 1)))

    ident.fitted.vals[i,1] <- fit.data.ident$sigmasq
    ident.fitted.vals[i,2] <- fit.data.ident$phi
}
remove("ident.increments", "incr", "fit.variog.ident", "fit.data.ident")
print("Estimated correlation lengths for identical twin OSSE:")
print(summary(ident.fitted.vals[,2]))
print(sd(ident.fitted.vals[,2]))
# space-separated default
write.matrix(ident.fitted.vals, "identical_gaussian_5_fits.dat")

plot(var2, scaled=TRUE, xlim=LIMITS.X, ylim=LIMITS.Y, xlab="Separation / km")
par(new=FALSE)
if (display.plots) {
    dev.copy2pdf(file=ident.variog.file)
} else {
    dev.off()
}

# Next, check how well it's able to recover the same correlation
# length assuming a different one
frat1.increments <- as.data.frame(
    scan("fraternal_gaussian_actual_5_assumed_2.csv", skip=1, sep=",",
         what=list(iteration=5, obs.time="", y=1., x=1., increment=.3))[c(1, 3, 4, 5)])

frat1.fitted.vals <- matrix(nrow=NRUNS, ncol=2)

frat1.variog.file <- "fraternal_actual_gaussian_5_assumed_gaussian_2_variogs.pdf"
if (! display.plots) {
    pdf(file=frat1.variog.file)
}


for (i in 1:NRUNS) {
    # Because 1-based indices
    run.start <- (i-1) * per.run + 1
    run.end <- run.start + per.run - 1
    incr <- as.geodata(as.data.frame(frat1.increments[run.start:run.end, 1:4]),
                       coords.col=c(2, 3), data.col=4, realisations=1)

    incr.var <- var(incr$data)
    # Get the variogram
    var2 <- variog(incr)
    plot(var2, scaled=TRUE, xlim=LIMITS.X, ylim=LIMITS.Y,
         bty="n", xaxt="n", yaxt="n", xlab="", ylab="")
    par(new=TRUE)
    # Fit a model to the variogram
    fit.variog.frat1 <- variofit(var2, cov.model="gneiting",
                                 ini.cov.pars=data.frame(
                                     sigmasq=incr.var,
                                     # This inversion assumed a
                                     # correlation length of 2
                                     phi=2),
                                 control=list(parscale=c(incr.var, 1, 1)))
    # Use the fitted models to start a restricted maximum likelihood
    # search using the original data
    fit.data.frat1 <- likfit(incr, ini.cov.pars=fit.variog.frat1,
                             cov.model="gneiting", lik.method="REML",
                             control=list(parscale=c(incr.var, 1)))

    frat1.fitted.vals[i,1] <- fit.data.frat1$sigmasq
    frat1.fitted.vals[i,2] <- fit.data.frat1$phi
}
remove("frat1.increments", "incr", "fit.variog.frat1", "fit.data.frat1")
print("Estimated correlation lengths for fraternal twin OSSE:")
print("Actual correlation length same as before (5),")
print("but inverted with half that (2)")
print(summary(frat1.fitted.vals[,2]))
print(sd(frat1.fitted.vals[,2]))
# space-separated default
write.matrix(ident.fitted.vals, "fraternal_actual_gaussian_5_assumed_gaussian_2_fits.dat")

plot(var2, scaled=TRUE, xlim=LIMITS.X, ylim=LIMITS.Y,
     xlab="Separation / km")
par(new=FALSE)
if (display.plots) {
    dev.copy2pdf(file=frat1.variog.file)
} else {
    dev.off()
}

# Next, check how well it's able to find a different correlation
# length assuming it's the same
frat2.increments <- as.data.frame(
    scan("fraternal_gaussian_actual_2_assumed_5.csv", skip=1, sep=",",
         what=list(iteration=5, obs.time="", y=1., x=1., increment=.3))[c(1, 3, 4, 5)])

frat2.fitted.vals <- matrix(nrow=NRUNS, ncol=2)

frat2.variog.file = "fraternal_actual_gaussian_2_assumed_gaussian_5_variogs.pdf"
if (! display.plots) {
    pdf(file=frat2.variog.file)
}

for (i in 1:NRUNS) {
    # Because 1-based indices
    run.start <- (i-1) * per.run + 1
    run.end <- run.start + per.run - 1
    incr <- as.geodata(as.data.frame(frat2.increments[run.start:run.end, 1:4]),
                       coords.col=c(2, 3), data.col=4, realisations=1)

    incr.var <- var(incr$data)
    # Get the variogram
    var2 <- variog(incr)
    plot(var2, scaled=TRUE, xlim=LIMITS.X, ylim=LIMITS.Y,
         bty="n", xaxt="n", yaxt="n", xlab="", ylab="")
    par(new=TRUE)
    # Fit a model to the variogram
    fit.variog.frat2 <- variofit(var2, cov.model="gneiting",
                                 ini.cov.pars=data.frame(sigmasq=incr.var,
                                                         phi=5),
                                 control=list(parscale=c(incr.var, 1, 1)))
    # Use the fitted models to start a restricted maximum likelihood
    # search using the original data
    fit.data.frat2 <- likfit(incr, ini.cov.pars=fit.variog.frat2,
                             cov.model="gneiting", lik.method="REML",
                             control=list(parscale=c(incr.var, 1)))

    frat2.fitted.vals[i,1] <- fit.data.frat2$sigmasq
    frat2.fitted.vals[i,2] <- fit.data.frat2$phi
}
remove("ident.increments", "incr", "fit.variog.frat2", "fit.data.frat2")
print("Estimated correlation lengths for fraternal twin OSSE:")
print("Actual correlation length smaller than before (2),")
print("but inverted with same (5)")
print(summary(frat1.fitted.vals[,2]))
print(sd(frat1.fitted.vals[,2]))
# space-separated default
write.matrix(ident.fitted.vals, "fraternal_actual_gaussian_2_assumed_gaussian_5_fits.dat")

plot(var2, scaled=TRUE, xlim=LIMITS.X, ylim=LIMITS.Y, xlab="Separation / km")
par(new=FALSE)
if (display.plots) {
    dev.copy2pdf(file=frat2.variog.file)
}

# Next, check how well it's able to find a different correlation
# length assuming it's the same
frat3.increments <- as.data.frame(
    scan("fraternal_actual_gaussian_5_assumed_exponential_5.csv", skip=1, sep=",",
         what=list(iteration=5, obs.time="", y=1., x=1., increment=.3))[c(1, 3, 4, 5)])

frat3.fitted.vals <- matrix(nrow=NRUNS, ncol=2)

frat3.variog.file <- "fraternal_actual_gaussian_5_assumed_exponential_5_variogs.pdf"
if (! display.plots) {
    pdf(file=frat3.variog.file)
}

for (i in 1:NRUNS) {
    # Because 1-based indices
    run.start <- (i-1) * per.run + 1
    run.end <- run.start + per.run - 1
    incr <- as.geodata(as.data.frame(frat3.increments[run.start:run.end, 1:4]),
                       coords.col=c(2, 3), data.col=4, realisations=1)

    incr.var <- var(incr$data)
    # Get the variogram
    var2 <- variog(incr)
    plot(var2, scaled=TRUE, xlim=LIMITS.X, ylim=LIMITS.Y,
         bty="n", xaxt="n", yaxt="n", xlab="", ylab="")
    par(new=TRUE)
    # Fit a model to the variogram
    fit.variog.frat3 <- variofit(var2, cov.model="exponential",
                                 ini.cov.pars=data.frame(sigmasq=incr.var,
                                                         phi=5),
                                 control=list(parscale=c(incr.var, 1, 1)))
    # Use the fitted models to start a restricted maximum likelihood
    # search using the original data
    fit.data.frat3 <- likfit(incr, ini.cov.pars=fit.variog.frat3,
                             cov.model="exponential", lik.method="REML",
                             control=list(parscale=c(incr.var, 1)))

    frat3.fitted.vals[i,1] <- fit.data.frat3$sigmasq
    frat3.fitted.vals[i,2] <- fit.data.frat3$phi
}
remove("ident.increments", "incr", "fit.variog.frat3", "fit.data.frat3")
print("Estimated correlation lengths for fraternal twin OSSE:")
print("Actual correlation gaussian")
print("but inverted with exponential")
print(summary(frat1.fitted.vals[,2]))
print(sd(frat1.fitted.vals[,2]))
# space-separated default
write.matrix(ident.fitted.vals, "fraternal_actual_gaussian_5_assumed_exponential_5_fits.dat")

plot(var2, scaled=TRUE, xlim=LIMITS.X, ylim=LIMITS.Y, xlab="Separation / km")
par(new=FALSE)
if (display.plots) {
    dev.copy2pdf(file=frat3.variog.file)
} else {
    dev.off()
}


# Next, check if it can see if the correlation function is different
frat4.increments <- as.data.frame(
    scan("fraternal_actual_exponential_5_assumed_gaussian_5.csv", skip=1, sep=",",
         what=list(iteration=5, obs.time="", y=1., x=1., increment=.3))[c(1, 3, 4, 5)])

frat4.fitted.vals <- matrix(nrow=NRUNS, ncol=2)

frat4.variog.file <- "fraternal_actual_exponential_5_assumed_gaussian_5_variogs.pdf"
if (! display.plots) {
    pdf(file=frat4.variog.file)
}

for (i in 1:NRUNS) {
    # Because 1-based indices
    run.start <- (i-1) * per.run + 1
    run.end <- run.start + per.run - 1
    incr <- as.geodata(as.data.frame(frat4.increments[run.start:run.end, 1:4]),
                       coords.col=c(2, 3), data.col=4, realisations=1)

    incr.var <- var(incr$data)
    # Get the variogram
    var2 <- variog(incr)
    plot(var2, scaled=TRUE, xlim=LIMITS.X, ylim=LIMITS.Y,
         bty="n", xaxt="n", yaxt="n", xlab="", ylab="")
    par(new=TRUE)
    # Fit a model to the variogram
    fit.variog.frat4 <- variofit(var2, cov.model="gneiting",
                                 ini.cov.pars=data.frame(sigmasq=incr.var,
                                                         phi=5),
                                 control=list(parscale=c(incr.var, 1, 1)))
    # Use the fitted models to start a restricted maximum likelihood
    # search using the original data
    fit.data.frat4 <- likfit(incr, ini.cov.pars=fit.variog.frat4,
                             cov.model="gneiting", lik.method="REML",
                             control=list(parscale=c(incr.var, 1)))

    frat4.fitted.vals[i,1] <- fit.data.frat4$sigmasq
    frat4.fitted.vals[i,2] <- fit.data.frat4$phi
}
remove("ident.increments", "incr", "fit.variog.frat4", "fit.data.frat4")
print("Estimated correlation lengths for fraternal twin OSSE:")
print("Actual correlation exponential")
print("but inverted with gaussian")
print(summary(frat1.fitted.vals[,2]))
print(sd(frat1.fitted.vals[,2]))
# space-separated default
write.matrix(ident.fitted.vals, "fraternal_actual_exponential_5_assumed_gaussian_5_fits.dat")

plot(var2, scaled=TRUE, xlim=LIMITS.X, ylim=LIMITS.Y, xlab="Separation / km")
par(new=FALSE)
if (display.plots) {
    dev.copy2pdf(file=frat4.variog.file)
} else {
    dev.off()
}
