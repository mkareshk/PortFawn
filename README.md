# project_finance


### Economic Model

```python
em = EconomicModel(risk_free_rate=0.02, annualized_days=252)
```


### Quantum Optimization Model

qom = QuantumOptModel(objective='BMOP', backend='qpu', annealing_time=100,)


### Classic Optimization Model

```python
com = ClassicOptModel(objective='MSRP', economic_model=em, target_return=0.1,
    target_sd=0.05,
    weight_bound=(0.05, 0.95)
    )
```
