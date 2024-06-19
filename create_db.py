import random
from typing import Any
import pandas as pd
from tqdm import trange

FROM_NUMBER_TO_VALUE = [
    "Asso",
    "Due",
    "Tre",
    "Quattro",
    "Cinque",
    "Sei",
    "Sette",
    "Fante",
    "Cavallo",
    "Re"
]

FROM_NUMBER_TO_SEGNO = [
    "Bastoni",
    "Spade",
    "Coppe",
    "Denari" # "Ori" :)
]

NUM_CARTE = 40

vittorie = 0
sconfitte = 0

class Carta:
    def __init__(self, id: int):
        self.id = id
        self.valore = FROM_NUMBER_TO_VALUE[self.id % 10]
        self.segno = FROM_NUMBER_TO_SEGNO[self.id // 10]

    def __repr__(self):
        return f"{self.valore} di {self.segno}"

class Giocatore:
    def __init__(self, id: int):
        self.id = id
        self.carta = None

    def ricevi_carta(self, carta: Carta):
        self.carta = carta

    def indovina(self, carte_avversari: list[Carta], num_giocatori: int):
        max_valore_avversari = max(c.id for c in carte_avversari)
        #scelta = random.choice([True, False]) # TODO
        scelta = NUM_CARTE - max_valore_avversari - 1 > max_valore_avversari - num_giocatori
        assert self.carta is not None
        return scelta, ((self.carta.id > max_valore_avversari) == scelta), self.carta.id > max_valore_avversari

def distribuisci_carte(num_giocatori: int):
    mazzo = [Carta(valore) for valore in range(NUM_CARTE)]
    random.shuffle(mazzo)
    return mazzo[:num_giocatori]

def gioca_partita(num_giocatori: int):
    global vittorie, sconfitte

    giocatori = [Giocatore(id) for id in range(num_giocatori)]
    carte = distribuisci_carte(num_giocatori)

    for giocatore, carta in zip(giocatori, carte):
        giocatore.ricevi_carta(carta)

    risultati: list[dict[str, Any]] = []
    k = []
    for giocatore in giocatori:
        carte_avversari = [g.carta for g in giocatori if g != giocatore]
        vinco, indovinato, esito = giocatore.indovina(carte_avversari, num_giocatori) # type: ignore
        assert giocatore.carta is not None
        risultati.append({
            'giocatore': giocatore.id,
            'carta': giocatore.carta.id,
            'indovinato': indovinato,
            'vinco': vinco,
            'k': k.copy(),
            'esito': esito
        })
        k.append(vinco)
        if indovinato:
            vittorie += 1
        else:
            sconfitte += 1

    return risultati

def crea_database_partite(num_partite: int, num_giocatori: int):
    global per_ogni_giocatore_quante_sconfitte
    global per_ogni_giocatore_quante_vittorie
    partite: list[list[dict[str, Any]]] = [[] for _ in range(num_giocatori)]
    for i in trange(num_partite, desc="Creazione database partite"):
        risultati = gioca_partita(num_giocatori)
        # [ { 'z': [12, 30, 2], 'k': [1, 1, -1], 'giocatore': 3, 'carta': 13, 'vinco': False, 'indovinato': True }, ... ]
        # 'z' indica le carte degli avversari
        # 'k' indica cosa han detto i giocatori prima di me
        # 'giocatore' indica il numero del turno
        # [ 'k', 'k', 'e' ]
        # 'e' indica se alla fine ho preso la mano (True) o no (False)
        for ris in risultati:
            if ris['esito'] == False and per_ogni_giocatore_quante_sconfitte[ris['giocatore']] >= per_ogni_giocatore_quante_vittorie[ris['giocatore']]:
                continue
            partite[ris['giocatore']].append({
                'z': [r['carta'] for r in risultati if r != ris],
                'k': ris['k'],
                'e': ris['esito'],
            })
            if ris['esito']:
                per_ogni_giocatore_quante_vittorie[ris['giocatore']] += 1
            else:
                per_ogni_giocatore_quante_sconfitte[ris['giocatore']] += 1
    return [pd.DataFrame(esiti) for esiti in partite]

if __name__ == "__main__":
    num_partite = 1000000
    num_giocatori = 4
    per_ogni_giocatore_quante_sconfitte = [0 for _ in range(num_giocatori)]
    per_ogni_giocatore_quante_vittorie = [0 for _ in range(num_giocatori)]
    list_of_df_partite = crea_database_partite(num_partite, num_giocatori)
    for i in range(len(list_of_df_partite)):
        df = list_of_df_partite[i]
        name = f"db_{num_giocatori}_{i}.csv"
        df.to_csv(f"dbs/{name}", index=False)
        print(f"Database salvato come {name}")
    print(f"Perc vittorie: {(vittorie / (vittorie + sconfitte))*100}%")

