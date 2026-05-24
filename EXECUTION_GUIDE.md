# Guía de Ejecución - Gravitar DQN & PPO (Challenge 1 & 3)

Esta guía proporciona los pasos y comandos exactos para ejecutar los agentes DQN y PPO en ALE/Gravitar-v5.

---

## Paso 1: Instalación de Dependencias

```bash
# Crear entorno virtual
python -m venv venv_gravitar

# Activar entorno virtual
# Windows:
venv_gravitar\Scripts\activate
# Linux/Kali:
source venv_gravitar/bin/activate

# Instalar dependencias
pip install stable-baselines3 gymnasium[atari] ale-py opencv-python torch tensorboard
```

---

## Paso 2: Prueba Rápida DQN (1 minuto)

Verifica que la instalación funciona correctamente entrenando DQN por 10,000 pasos.

```bash
python gravitar_dqn.py --mode train --model-path models/test_dqn --timesteps 10000
```

**Salida esperada**: El modelo se guarda en `models/test_dqn.zip` y logs en `logs/gravitar_dqn/`.

---

## Paso 3: Prueba Rápida PPO (1 minuto)

Verifica que la instalación funciona correctamente entrenando PPO por 10,000 pasos.

```bash
cd challenge3/group4
python gravitar_ppo.py --mode train --model-path ../../models/test_ppo --timesteps 10000
cd ../..
```

**Salida esperada**: El modelo se guarda en `models/test_ppo.zip` y logs en `logs/gravitar_ppo/`.

---

## Paso 4: Entrenamiento DQN Completo (Challenge 1)

**Nota**: Si ya completaste el Challenge 1, puedes saltar este paso.

### Opción A: Entrenar configuración individual

```bash
python gravitar_dqn.py --mode train --model-path models/gravitar_dqn_g4
```

- **Tiempo estimado**: ~12-18h CPU / ~3-4h GPU
- **Pasos**: 300,000
- **Modelo**: `models/gravitar_dqn_g4.zip`

### Opción B: Ejecutar sweep completo (recomendado)

```bash
python gravitar_dqn.py --mode sweep --sweep-file sweep_configs.json --model-path models/gravitar_dqn_best
```

- **Tiempo estimado**: ~36-54h CPU / ~9-12h GPU
- **Configuraciones**: 5
- **Semillas**: 3 por configuración (42, 43, 44)
- **Total runs**: 15
- **Modelo**: `models/gravitar_dqn_best.zip` (mejor configuración)

---

## Paso 5: Entrenamiento PPO Completo (Challenge 3)

Este es el paso principal del Challenge 3.

### Opción A: Entrenar configuración individual

```bash
cd challenge3/group4
python gravitar_ppo.py --mode train --model-path ../../models/gravitar_ppo_g4
cd ../..
```

- **Tiempo estimado**: ~12-18h CPU / ~3-4h GPU
- **Pasos**: 5,000,000
- **Modelo**: `models/gravitar_ppo_g4.zip`

### Opción B: Ejecutar sweep completo (recomendado para Challenge 3)

```bash
cd challenge3/group4
python gravitar_ppo.py --mode sweep --sweep-file sweep_configs_ppo.json --model-path ../../models/gravitar_ppo_best
cd ../..
```

- **Tiempo estimado**: ~36-54h CPU / ~9-12h GPU
- **Configuraciones**: 9
- **Semillas**: 3 por configuración (42, 43, 44)
- **Total runs**: 27
- **Modelo**: `models/gravitar_ppo_best.zip` (mejor configuración)

---

## Paso 6: Monitorear Entrenamiento con TensorBoard

Abre terminales separadas para monitorear ambos algoritmos simultáneamente.

### Terminal 1 - Monitorear DQN

```bash
tensorboard --logdir logs/gravitar_dqn/sweep --port 6006
```

### Terminal 2 - Monitorear PPO

```bash
tensorboard --logdir logs/gravitar_ppo/sweep --port 6007
```

**Abre en tu navegador**:
- DQN: http://localhost:6006
- PPO: http://localhost:6007

**Métricas disponibles**:
- `rollout/ep_rew_mean`: Recompensa promedio (últimos 100 episodios)
- `training/episode_reward`: Recompensa por episodio
- `train/loss`: Pérdida de entrenamiento
- `train/learning_rate`: Learning rate actual

---

## Paso 7: Ver Agente Entrenado Jugar

### Ver DQN

```bash
python gravitar_dqn.py --mode play --model-path models/gravitar_dqn_best --episodes 5
```

### Ver PPO

```bash
cd challenge3/group4
python gravitar_ppo.py --mode play --model-path ../../models/gravitar_ppo_best --episodes 5
cd ../..
```

**Nota**: Requiere display/X11 para renderizar la ventana del juego.

---

## Paso 8: Inspeccionar Hiperparámetros

### Inspeccionar DQN

```bash
python gravitar_dqn.py --mode inspect --model-path models/gravitar_dqn_best
```

### Inspeccionar PPO

```bash
cd challenge3/group4
python gravitar_ppo.py --mode inspect --model-path ../../models/gravitar_ppo_best
cd ../..
```

**Salida**: Muestra todos los hiperparámetros del modelo guardado.

---

## Resumen de Archivos Generados

### Modelos
- `models/test_dqn.zip` - Prueba DQN
- `models/test_ppo.zip` - Prueba PPO
- `models/gravitar_dqn_best.zip` - Mejor modelo DQN
- `models/gravitar_ppo_best.zip` - Mejor modelo PPO

### Logs
- `logs/gravitar_dqn/sweep/` - Logs DQN (por configuración y semilla)
- `logs/gravitar_ppo/sweep/` - Logs PPO (por configuración y semilla)

---

## Configuraciones de Hiperparámetros

### DQN (sweep_configs.json)
5 configuraciones variando learning rate, buffer size, y batch size.

### PPO (sweep_configs_ppo.json)
9 configuraciones variando sistemáticamente:
- Learning rate: 1e-4, 2.5e-4, 5e-4
- Horizon (n_steps): 512, 1024, 2048
- Entropy coefficient: 0.001, 0.01, 0.02

---

## Tiempos de Ejecución Estimados

| Tarea | CPU | GPU |
|-------|-----|-----|
| Prueba rápida (10k pasos) | 1 min | 30 seg |
| Entrenamiento individual DQN (300k) | 12-18h | 3-4h |
| Entrenamiento individual PPO (5M) | 12-18h | 3-4h |
| Sweep DQN (5 configs × 3 seeds) | 36-54h | 9-12h |
| Sweep PPO (9 configs × 3 seeds) | 36-54h | 9-12h |

---

## Solución de Problemas

### Error: "No module named gymnasium"
```bash
pip install gymnasium[atari]
```

### Error: "No module named stable_baselines3"
```bash
pip install stable-baselines3
```

### Error: "ALE not found"
```bash
pip install ale-py
```

### Error: "CUDA out of memory"
- Reduce `batch_size` en el archivo JSON de configuraciones
- O usa CPU en lugar de GPU

### Error: "Model not found"
- Asegúrate de haber ejecutado el entrenamiento primero
- Verifica que la ruta del modelo sea correcta

---

## Checklist para Challenge 3

- [ ] Instalar dependencias
- [ ] Ejecutar prueba rápida PPO
- [ ] Ejecutar sweep PPO completo (9 configs × 3 seeds)
- [ ] Verificar logs en TensorBoard
- [ ] Verificar que el mejor modelo se guardó en `models/gravitar_ppo_best.zip`
- [ ] Llenar CHECKLIST.md con resultados
- [ ] Completar análisis comparativo en `comparative_analysis_dqn_vs_ppo.md`
- [ ] Completar artículo IEEE en `ieee_paper_dqn_to_ppo_gravitar.md`

---

## Contacto

Para preguntas sobre la ejecución, consulta:
- `README.md` - Documentación completa
- `challenge3/group4/CHECKLIST.md` - Comandos específicos Challenge 3
- `challenge3/group4/Challenge3.md` - Requisitos del Challenge 3
