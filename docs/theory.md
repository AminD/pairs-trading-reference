# Theory notes

## 1. Stationarity and random walks

A series is **stationary** when its mean and variance do not change over time; it keeps returning to its mean. A **random walk**, $p_t = p_{t-1} + \varepsilon_t$, is not: its variance grows with time and it has no level to return to. Log prices of stocks are close to random walks, which is why their levels cannot be forecast.

Regressing one random walk on another, unrelated one produces a high $R^2$ and a "significant" slope most of the time. This is the **spurious regression** problem (Granger and Newbold, 1974): correlation between the levels of trending series means nothing.

## 2. Cointegration

Two non-stationary series $\log Y_t$ and $\log X_t$ are cointegrated if there is a $\beta$ such that

$$s_t = \log Y_t - \beta\,\log X_t - \alpha$$

is stationary. Economically, a common factor drives both prices, and $\beta$ is the ratio in which they load on it. The spread $s_t$ is what a long-short position of 1 in $Y$ and $-\beta$ in $X$ earns (in logs), so a stationary spread is a tradable, mean-reverting quantity.

## 3. The Engle-Granger test

**Step 1.** Estimate $\beta$ and $\alpha$ by least squares and form the residual $\hat s_t$.

**Step 2.** Test $\hat s_t$ for a unit root with an augmented Dickey-Fuller regression,

$$\Delta \hat s_t = \gamma\,\hat s_{t-1} + \sum_{i=1}^{p}\varphi_i\,\Delta \hat s_{t-i} + \varepsilon_t .$$

Under the null hypothesis of no cointegration, $\gamma = 0$. The statistic is $t = \hat\gamma / \text{se}(\hat\gamma)$, and large negative values reject the null. The lagged differences absorb the autocorrelation of the residual.

Two things make the reference distribution non-standard. Under the null the regressor $\hat s_{t-1}$ is itself a random walk, so $t$ follows a Dickey-Fuller distribution, not a Student one. And $\hat s_t$ is a regression residual: least squares has chosen the $\beta$ that makes it look as stationary as possible, which shifts the distribution further. MacKinnon's response surfaces give the p-values for this case.

The test is not symmetric in $Y$ and $X$. Running it both ways and keeping the better result raises the false-positive rate from 5% to about 9% (measured by simulation here).

## 4. Half-life of mean reversion

For a stationary AR(1) spread, $s_t - \bar s = \phi\,(s_{t-1} - \bar s) + \varepsilon_t$, a deviation is multiplied by $\phi$ each day and halves after

$$h = -\frac{\ln 2}{\ln \phi}$$

days. In continuous time this is the Ornstein-Uhlenbeck process $ds = -\kappa(s - \bar s)\,dt + \sigma\,dW$, with $\phi = e^{-\kappa\Delta t}$ and $h = \ln 2/\kappa$. The half-life sets the horizon of a trade: it must be short relative to the trading period and long relative to one day.

## 5. Multiple testing

With $m$ independent tests at level 5% and no true effect, $0.05\,m$ tests reject on average. For 567 pairs that is 28 pairs, or 52 with the two-direction scan.

The **Benjamini-Hochberg** procedure controls the false discovery rate, the expected share of false positives among the rejections. Sort the p-values, $p_{(1)} \le \dots \le p_{(m)}$, find the largest $k$ with

$$p_{(k)} \le \frac{k}{m}\,q ,$$

and reject the $k$ smallest. With $q = 5\%$, at most 5% of the pairs kept are expected to be false discoveries.

## 6. Kalman filter for the hedge ratio

State-space model with hidden state $\theta_t = (\beta_t, \alpha_t)^\top$:

$$\theta_t = \theta_{t-1} + w_t,\quad w_t \sim \mathcal{N}(0, W), \qquad \log Y_t = H_t\,\theta_t + v_t,\quad v_t \sim \mathcal{N}(0, R),$$

with $H_t = (\log X_t,\ 1)$. Given the estimate $\hat\theta_{t-1}$ and its covariance $P_{t-1}$:

| Step | Formula |
|---|---|
| Predict | $P_{t\mid t-1} = P_{t-1} + W$ |
| Forecast error | $e_t = \log Y_t - H_t\hat\theta_{t-1}$ |
| Its variance | $Q_t = H_t P_{t\mid t-1} H_t^\top + R$ |
| Gain | $K_t = P_{t\mid t-1} H_t^\top / Q_t$ |
| Update | $\hat\theta_t = \hat\theta_{t-1} + K_t e_t$, $\quad P_t = P_{t\mid t-1} - K_t H_t P_{t\mid t-1}$ |

$e_t$ and $Q_t$ depend only on data up to $t-1$ and on $\log X_t$, so $z_t = e_t/\sqrt{Q_t}$ is a signal without look-ahead, and $\hat\beta_{t-1}$ is the hedge ratio to use on day $t$.

With $W = 0$ the filter is recursive least squares and converges to the regression estimate. With $W > 0$ it forgets old data. Here $W = (R/D^2)\,I$: the steady-state gain on the intercept is about $1/D$, so the filter absorbs a persistent deviation in roughly $D$ days. If $D$ is not well above the half-life of the spread, the filter treats every deviation as a change of the relation and removes the signal.

## 7. Trading rule and returns

With $z_t$ the standardised spread: open a short position in the spread when $z_t > z_{\text{in}}$ and a long one when $z_t < -z_{\text{in}}$; close when $|z_t| < z_{\text{out}}$; stop when $|z_t| > z_{\text{stop}}$.

A unit long position holds $w_Y = 1/(1+|\beta|)$ of $Y$ and $w_X = -\beta/(1+|\beta|)$ of $X$, so that $|w_Y| + |w_X| = 1$. Weights chosen at the close of day $t$ earn

$$r_{t+1} = w_{Y,t}\,r^Y_{t+1} + w_{X,t}\,r^X_{t+1} - c\,\big(|\Delta w_{Y,t}| + |\Delta w_{X,t}|\big),$$

with $c$ the proportional cost.

## 8. How sure is a Sharpe ratio?

For returns that are roughly independent, the annualised Sharpe ratio $S$ estimated over $n$ years has standard error (Lo, 2002)

$$\text{se}(S) \approx \sqrt{\frac{1 + S^2/2}{n}} .$$

Over 15 years this is about 0.26 for a true Sharpe ratio of zero: any estimate below 0.5 in absolute value is indistinguishable from nothing. Trying several strategies and variants makes the bar higher still, since the best of many noisy estimates is biased upwards.
