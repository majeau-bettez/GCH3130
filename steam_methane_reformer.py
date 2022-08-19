#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code créé pour le cours de GCH3110 de Polytechnique Montréal - session d'été 2022

@author: Guillaume Majeau-Bettez
@author: Alexandre Sévigny
@author: Symphorien Notue
"""

import warnings
import numpy as np
import pandas as pd
import scipy.integrate as spi
import matplotlib.pyplot as plt
import os
import contextlib
import copy

# Constants
R = 8.314                                                                                                # kJ / (kmol.K)

class IndexSMR:
    """ Une simple classe pour gérer les indexes. 

    Les substances, la pression et la température doivent toujours être dans le même ordre dans les vecteurs. Cette
    classe spécifie cet ordre et aide à  s'assurer qu'on ne se trompe pas.
    """

    def __init__(self):
        self.H2O = 0
        self.CO2 = 1
        self.H2 = 2
        self.CO = 3
        self.CH4 = 4
        self.p = 5
        self.T = 6

    @property
    def subst(self):
        return ['H2O', 'CO2', 'H2', 'CO', 'CH4']

    @property
    def aslist(self):
        return ['H2O', 'CO2', 'H2', 'CO', 'CH4', 'p', 'T']

    @property
    def rx(self):
        return [0, 1, 2]


# Définir les indexes comme variable  globale
ix = IndexSMR()


# noinspection PyPep8Naming
class SteamMethaneReformer:
    """ Classe qui génère un Steam Methane Reformer. Contient les données, les équations, et gère leur manipulation.
    """

    def __init__(self, data_path):
        """ Initialise l'objet Steam Methane Reformer, et ses données initiales

        Parameters
        ----------
        data_path : string
            Pointe vers l'endroit où se trouve le fichier excel avec les données

        """

        # Initialisation des paramètres d'expérimentation (vides)
        self.F0_tot = 0.0  # Débit molaire total entrant                                                     # kmol / h
        self.P0_tot = 0.0  # Pression totale initiale                                                             # bar
        self.T0 = 0.0  # Température initiale des réactifs                                                          # K
        self.Ta = 0.0  # Température des parois du réacteur                                                         # K

        # Read additional data
        with pd.ExcelFile(data_path) as f:

            # Lire les propriétés des substances
            raw_subst = pd.read_excel(data_path, 'prop_subst', index_col=0).fillna(0.0)
            x0, Ka0, DeltaHa, M, _ = [raw_subst[c] for c in raw_subst.columns]

            self.x0 = x0[ix.subst].values  # Les fractions molaires des substance                        # adimensionnel
            self.Ka0 = Ka0[ix.subst].values  # Les constantes d'adsorption
            self.DeltaHa = DeltaHa[ix.subst].values  # L'enthalpie d'adsorption                               # kJ / mol
            self.M = M[ix.subst].values  # Masses molaires                                                   # kJ / kmol

            # Lire les caractéristiques des réactions
            raw_rx = pd.read_excel(f, 'prop_rx', index_col=0).fillna(0.0)
            k0, E, DeltaH0 = [raw_rx[c] for c in raw_rx.columns]
            #
            self.k0 = k0.values  # Constantes de réactions                               #mol bar^[.5 | -1] kg_cat⁻¹ s⁻¹
            self.E = E.values  # Énergies d'activation                                                        # kJ / mol
            self.DeltaH0 = DeltaH0.values  # Enthalpies de réaction standards                                 # kJ / mol

            # Lire les coefficients stoechiométriques normalisés
            self.nu = pd.read_excel(f, 'nu', index_col=0).fillna(0.0)

        # PROPRIÉTÉS DU FLUIDE
        self.mu0 = 238E-7  # voir calculs du rapport                                                            # Pa * s

        # PROPRIÉTÉS DU RÉACTEUR
        #
        self.U = 454.25   # Refroidissement                                                                  W / (m^2.K)
        self.r_in = 0.06   # Diamètre interne                                                                       # m
        self.L = 12.5      # Longueur réacteur                                                                       # m
        self.a = 2 / self.r_in  # aires spécifique                                                                  m^-1
        self.Dp = 5.4E-3  # Diamètre particule                                                                        m
        self.Ac = np.pi * self.r_in**2  # Aire de la coupe transversale                                               m2
        self.rho_b = 1100  # Densité "bulk" du lit catalytique                                      # [kg / m^3 de tube]
        self.phi = 0.605  # porosité du lit catalytique                                                    [sans unités]
        self.rho_c = self.rho_b / (1 - self.phi)  # densité du catalyseur                                        kg / m3

        self.Wmax = self.rho_b * self.Ac * self.L  # Capacité de catalyseur dans le tube                            # kg

        # Initialiser des variables vides
        self.alpha = 0  # Paramètre pour la perte de charge
        self.beta0 = 0  # Paramètre pour la perte de charge
        self.sol = None    # Solutions à la résolution d'un système d'équations différentielles
        self.y_out = None  # Résultats du système d'équation différentielles

        # Indexes:
        self.ix = ix

    # ==================================================================================================================
    # Propriétés à être recalculées 'on the spot' quand on les appelle.
    # ==================================================================================================================

    @property
    def F0(self):
        """ Débit molaire entrant pour chaque réactif, en kmol / h """
        return self.F0_tot * self.x0  # Débit molaire entrant pour chaque réactif                             # kmol / h

    @property
    def X(self):
        """ Conversion du méthane"""
        return (self.F0[ix.CH4] - self.y_out['CH4']) / self.F0[ix.CH4]

    @property
    def rendement(self):
        """ Rendement de l'hydrogène à chaque point du réacteur, mol_H2 par mol_CH4"""
        return self.y_out['H2'] / self.F0[ix.CH4]

    @property
    def X_fin(self):
        """ Conversions finales pour chacun des réactifs"""
        X =  (self.F0 - self.F) / self.F0
        return X[X>0]

    @property
    def rendement_fin(self):
        """ Rendement final en moles d'hydrogène par mole de méthane"""
        return self.F['H2'] / self.F0[ix.CH4]

    # Conditions initiales
    @property
    def y0(self):
        """ Les valeurs initiales de toutes les variables à optimiser. Les débits molaires F0, p, et T0"""
        params = np.array([1,  # Pression adimensionnelle, par définition = 1
                           self.T0
                           ])
        return np.concatenate([self.F0, params])  # TODO: check formatting

    @property
    def F(self):
        """ Débit molaires à la sortie du réacteur"""
        return self.y_out.iloc[-1][ix.subst]

    # ==================================================================================================================
    # Le système d'équation différentielles... le coeur du modèle
    # ==================================================================================================================

    def system_ode(self, W, y0):
        """ Système d'équations différentielles à résoudre pour notre système SMR

        Parameters
        ----------
        W : float
            Variable indépendante, masse de catalyseur en kg
        y0 : 1D numpy array
            Valeurs initiales des variables dépendantes, dans l'ordre défini par la classe IndexSMR
        """


        # Initialisation
        dYdW = np.zeros(len(y0))

        # Organiser préliminaire des données pour une manipulation plus facile
        F = y0[:5]    # débits
        p = y0[ix.p]  # pression
        T = y0[ix.T]  # Température

        # Calcul du débit molaire total
        F_tot = F.sum()                                                                                      # kmol / h

        # Calcul des pressions partielles de chaque substance
        P = p * self.P0_tot * F / F_tot                                                                           # bar

        # Calcul des constantes de réactions
        k = self.k0 * 1000 * np.exp(-self.E * 1e3 / (R * T))                         # mol / (h * bar^[0.5|-1] * kg_cat)

        # Calcul des constantes d'adsorption
        Ka = self.Ka0 * np.exp(-self.DeltaHa * 1000 / (R * T))

        # Calcul des constates d'équilibre
        K = np.zeros(len(ix.rx))
        K[0] = np.exp(-26830 / T + 30.144)
        K[1] = np.exp(4400 / T - 4.036)
        K[2] = K[0] * K[1]

        # Calcul des vitesses de réaction
        # mols / h.kg_cat
        r = np.zeros(3)
        omega = 1 + Ka[ix.CO] * P[ix.CO] + Ka[ix.H2] * P[ix.H2] + \
            Ka[ix.CH4] * P[ix.CH4] + (Ka[ix.H2O] * P[ix.H2O] / P[ix.H2])
        r[0] = (k[0] / P[ix.H2]**2.5) * (P[ix.CH4] * P[ix.H2O] - P[ix.H2]**3 * P[ix.CO] / K[0]) * omega**-2
        r[1] = (k[1] / P[ix.H2]) * (P[ix.CO] * P[ix.H2O] - P[ix.H2] * P[ix.CO2] / K[1]) * omega**-2
        r[2] = (k[2] / P[ix.H2]**3.5) * (P[ix.CH4] * P[ix.H2O]**2 - P[ix.H2]**4 * P[ix.CO2] / K[2]) * omega**-2

        # On passe en kmol / h.kg_cat
        r = r / 1000  # 1000

        # Calcul des capacités calorifiques à T
        # Cp [kJ/kmol.K]
        Cp = calc_cp(T)

        # Calcul des enthalpies de réaction
        # kJ / kmol
        DeltaHrx = calc_DeltaHrx(T)

        # Ajuster les unités pour les vitesses de réaction
        # (kmol / h.kg_cat) * (1h / 3600s) = kmol / kg_cat.s
        r_a = -r / 3600

        # ========================================================================
        # Équation d'énergie (voir T11-1.F in p499 & eq 12-35 p581)
        #
        # (kmol * kg_cat^-1 * s^-1 ) ( kJ * kmol^-1) = kW/kg_cat
        Q_g = r_a @ DeltaHrx

        # W / m2.K * m⁻¹ * (m3 / kg_cat) * (1 kW / 1000W) = kW/kg_cat
        Q_r = self.U * self.a / self.rho_b * (T - self.Ta) / 1000

        ###############################################################################################################
        #    Enfin les équations différentielles
        ###############################################################################################################
        #
        # Équation de température
        #   Unités:
        #                   kW kg_cat^-1
        #    --------------------------------------------------  = K / kg_cat
        #     (kmol / h) (1 h / 3600s) * (kJ * kmol^-1 * K^-1)
        #
        #
        dYdW[ix.T] = (Q_g - Q_r) / ((F / 3600) @ Cp)


        # Équations de débits molaires
        # dF_j/dW  [kmol / h.kg_cat]
        dYdW[ix.CH4] = -r[0] - r[2]
        dYdW[ix.H2O] = -r[0] - r[1] - 2 * r[2]
        dYdW[ix.CO] = r[0] - r[1]
        dYdW[ix.CO2] = r[1] + r[2]
        dYdW[ix.H2] = 3 * r[0] + r[1] + 4 * r[2]

        # Équation de température
        dYdW[ix.p] = (- self.alpha / (2 * p)) * (T / self.T0) * (F_tot / self.F0_tot)

        return dYdW

    def calc_ergun(self, T, F, P):
        """" Calcule des paramètres alpha et beta pour le la perte de charge suivant l'équation d'Ergun

        Parameters
        ----------
        T : float
            Température [K]
        F : array
            Débit molaire [kmol / h]
        P : float
            Pression [bar]
        """
        # changement d'unités à l'interne
        F = F / 3600                                                                                           # kmol /s
        F_tot = F.sum()                                                                                       # kmol / s
        P = P * 1e5                                                                                                 # Pa
        gc = 1  # en métrique OK

        # Masse molaire moyenne
        M0_av = (F @ self.M) / F_tot                                                                     # kg / kmol_tot

        # Débit massique superficiel à l'entrée
        G = M0_av * F_tot / self.Ac                                                                      # kg / (m2 * s)

        # densité initiale du gaz
        rho0 = P * M0_av / (R * T) / 1000                                                                      # kg / m3

        self.beta0 = G / (rho0 * gc * self.Dp) * ((1 - self.phi) / self.phi**3) * \
                     (150 * (1 - self.phi) * self.mu0 / self.Dp + 1.75 * G)  # J
        self.alpha = 2 * self.beta0 / (self.Ac * self.rho_c * (1 - self.phi) * P)                                # 1/kg

    # ==================================================================================================================
    # Routines de résolutions
    # ==================================================================================================================

    def solve(self, F0_tot, P0_tot, T0, Ta, verbose=True):
        """ Résoudre le système d'équation différentielle

        Parameters
        ----------
        F0_tot : float
            Débit molaire entrant total, toute substance confondue [kJ / mol]
        P0_tot : float
            Pression initiale totale [bar]
        T0 : float
            Température initiale [K]
        Ta : float
            Température de paroi [K]
        verbose : boolean (default=True)
            Si False, empêche l'impression de messages d'avertissement ou d'échecs du solver pendant la résolution


        Returns
        -------
        None, mais calcule l'attribut y_out qui contient l'évolution des variables dépendantes le long du réacteur

        """

        # Définir les paramètres d'expérimentation
        self.F0_tot = F0_tot
        self.P0_tot = P0_tot
        self.T0 = T0
        self.Ta = Ta

        self.calc_ergun(self.T0, self.F0, self.P0_tot)

        with warnings.catch_warnings():
            # Empêche les messages excessifs d'avertissement du solver
            warnings.filterwarnings('ignore')

            # =========================================================================================================
            # Le coeur de la résolution ici:
            self.sol = spi.solve_ivp(self.system_ode, t_span=(0, self.Wmax), y0=self.y0, method='LSODA')
            # =========================================================================================================

        # Finalisation des résultats
        self.y_out = pd.DataFrame(self.sol.y.T, columns=ix.aslist)

        # Rétroaction
        if self.sol.success:
            if verbose:
                print("Tout est beau!")
        else:
            if verbose:
                warnings.warn(self.sol.message, UserWarning)

    def solve_and_adjustF0(self, F0_tot, P0_tot, T0, Ta, verbose=True, FH2_base=8.6891, excess=0.1):
        """ Résoudre le système d'équations différentielles en ajustant le débit de réactifs entrant.

        Résout le système ODE, puis calcule l'excès de production de H2, puis réduit le débit molaire d'entrée
        proportionnellement à cet excès, puis re-résout le système ODE, jusqu'à convergence.

        Parameters
        ----------
        F0_tot : float
            Débit molaire entrant total, toute substance confondue [kJ / mol]
        P0_tot : float
            Pression initiale totale [bar]
        T0 : float
            Température initiale [K]
        Ta : float
            Température de paroi [K]
        verbose : boolean (default=True)
            Si False, empêche l'impression de messages d'avertissement ou d'échecs du solver pendant la résolution
        FH2_base: float
            Débit molaire du cas de base
        excess: float
            Excès relatif de production de H2 toléré.
        """

        if verbose:
            self._solve_and_adjustF0(F0_tot, P0_tot, T0, Ta, verbose, FH2_base, excess)
        else:
            with suppress_stdout_stderr():
                self._solve_and_adjustF0(F0_tot, P0_tot, T0, Ta, verbose, FH2_base, excess)

    def _solve_and_adjustF0(self, F0_tot, P0_tot, T0, Ta, verbose, FH2_base, excess):
        """ Méthode cachée, appelée soit en mode 'verbose' ou non par self.solve_and_adjustF0"""

        # Première ronde
        self.solve(F0_tot, P0_tot, T0, Ta, verbose)

        # Calculer l'excès relatif d'hydrogène
        overproduction_ratio = self.F['H2'] / FH2_base

        # Empêche Fortran d'imprimer des mises à jour excessives
        with contextlib.redirect_stdout(open(os.devnull, 'w')):

            # Re-résoudre et ré-ajuster jusqu'à convergence des débits d'hydrogène
            while overproduction_ratio > (1 + excess):

                F0_tot_backup = copy.copy(F0_tot)

                # New run with reduced F0_tot
                adjustment = 1 + (overproduction_ratio - 1) * 0.8
                F0_tot = F0_tot / adjustment

                # Re-résoudre
                self.solve(F0_tot, P0_tot, T0, Ta, verbose)

                # Revérifier la surproduction d'hydrogène
                overproduction_ratio = self.F['H2'] / FH2_base

                if overproduction_ratio < 1:
                    # Oups, on a "surcorrigé"... en déficit
                    self.solve(F0_tot_backup, P0_tot, T0, Ta, verbose)
                    assert (self.F['H2'] >= FH2_base)
                    break

    def explore_all_options(self, T0, Ta_range, P0_range, F0_tot, FH2_base, figname='influence_pT'):
        """ Teste le rendement et la conversion du réacteur pour un éventail de paramètres

        Calcule toutes les combinaisons de températures de parois Ta et de pressions initiales P0, en ajustant à chaque
        fois le débit molaire des réactifs afin d'obtenir un meilleur temps de résidence et approximativement un même
        débit de production d'hydrogène.

        Parameters
        ----------
        T0 : float
            Température initiale [K]
        Ta_range : iterable (list or array)
            Températures à considérer [K]
        P0_range : iterable (list or array)
            Pressions initiales à considérer
        F0_tot : float
            Débit molaire du cas de base
        FH2_base : float
            Débit de production d'hydrogène du cas de base
        figname : string
            texte à ajouter aux noms de fichier des figures générées

        Returns
        -------
        all_yields : DataFrame
            Rendements pour toutes les combinaisons de pressions et températures
        all_X : DataFrame
            Conversions pour toutes les combinaisons de pressions et températures

        """

        # Initialiser les variables
        all_yields = pd.DataFrame(index=Ta_range, columns=P0_range)
        all_X = pd.DataFrame(index=Ta_range, columns=P0_range)

        # Tester toutes les combinaisons
        for Ta in Ta_range:
            for P0_tot in P0_range:
                self.solve_and_adjustF0(F0_tot, P0_tot, T0, Ta, FH2_base=FH2_base, verbose=False)
                if self.sol.success and self.F['H2'] >= FH2_base:
                    all_yields.loc[Ta, P0_tot] = self.rendement_fin
                    all_X.loc[Ta, P0_tot] = self.X_fin['CH4']

        # finaliser les résultats
        all_yields = all_yields.dropna(axis=0, how='all').dropna(axis=1, how='all')
        all_X = all_X.dropna(axis=0, how='all').dropna(axis=1, how='all')

        # Illustrer les résultats
        for name, i in all_yields.iteritems():
            plt.plot(i, label=f"P={name} bar")
        plt.xlabel('Température de la paroi ($T_a$) [K]')
        plt.ylabel("Rendement (mol H$_2$ / mol CH$_4$)")
        plt.title("Influence sur le rendement de $T_a$ et $P_0$ \n (avec $F0_{tot}$ ajusté en chaque point pour un"
                  " même débit de $F_{H_2}$)")
        plt.grid()
        plt.legend()
        savefig(figname+'_rendement')
        plt.show()

        for name, i in all_X.iteritems():
            plt.plot(i, label=f"P={name} bar")
        plt.ylabel('X_CH4')
        plt.xlabel('T [K]')
        plt.title("Influence de $T_a$ et $P_0$ (avec $F0_{tot}$ ajusté) sur le taux de conversion")
        plt.grid()
        plt.legend()
        savefig(figname + '_conversion')
        plt.show()

        return all_yields, all_X

    # =================================================================================================================
    # Outils de visualisation
    # =================================================================================================================

    def plot(self, figname_prefix='', what='all'):
        """

        Parameters
        ----------
        figname_prefix : string
            Name to identify the SMR experiment  in the figure name
        what : string [default = 'all']
            Quoi représenter. Soit 'F', 'p, 'T', 'X', ou  'all'
        """
        fig = plt.figure()
        ax = plt.axes()

        if what == 'F' or what == 'all':
            for i in ix.subst:
                ax.plot(self.sol.t, self.y_out[i], label=i)
            plt.legend()
            plt.title(f"Débits molaires pour $T_a$={self.Ta} K, $P_0$={self.P0_tot} bar, "
                      "et $F_{{tot_0}}$ = {self.F0_tot:.1f} kmol/h")
            plt.xlabel('Progression dans le PBR [kg catalyseur]')
            plt.ylabel('F [kmol / h]')
            plt.grid()
            savefig(figname_prefix + 'debitsMolaires')
            plt.show()

        if what == 'p' or what == 'all':
            plt.title("Perte de charge relative le long du tube")
            plt.xlabel('Progression dans le PBR [kg catalyseur]')
            plt.ylabel('Pression [adimensionnelle]')
            plt.plot(self.sol.t, self.y_out['p'], label='p')
            plt.grid()
            savefig(figname_prefix + 'perteDeCharge')
            plt.show()

        if what == 'T' or what == 'all':
            plt.title(f"Profil température du milieu pour une température de paroi de {self.Ta} K")
            plt.plot(self.sol.t, self.y_out['T'], label='T milieu')
            plt.axhline(self.Ta, linestyle='dashed', label='T paroi')
            plt.legend()
            plt.grid()
            plt.ylabel('Température du milieu réactionnel [K]')
            plt.xlabel('Progression dans le PBR [kg catalyseur]')
            savefig(figname_prefix + 'profilDeTempérature')
            plt.show()

        if what == 'X' or what == 'all':
            plt.title(
                f"Profil de conversion pour $T_a$={self.Ta} K, $P_0$={self.P0_tot} bar,"
                " et $F0_{{tot}}$ = {self.F0_tot:.1f} kmol/h")
            plt.plot(self.sol.t, self.X, label="$X_{CH_4}$")
            plt.ylabel("Conversion")
            plt.xlabel('Progression dans le PBR [kg catalyseur]')
            plt.legend()
            plt.grid()
            savefig(figname_prefix + 'profilDeConversion')
            plt.show()

# =====================================================================================================================
# Sous-calculs
# =====================================================================================================================

def calc_DeltaHrx(T):
    """ Calcule les enthalpies de réaction en fonction de la température

    Parameters
    ----------
    T : float
        Température de réaction [K]

    Returns
    -------
    DeltaHrx : Numpy array (1-D)
        Les enthalpies de réaction
    """
    TR = 298  # Température de référence pour l'enthalpie de réaction                                            # K

    AH2 = 19.67099
    BH2 = 6.96815*10**-2
    CH2 = -2.0009*10**-4
    DH2 = 2.89492*10**-7
    EH2 = -2.22474*10**-10
    F_H2 = 8.81465*10**-14
    GH2 = -1.42043*10**-17

    ACH4 = 44.35658
    BCH4 = -0.14623
    CCH4 = 6.00245*10**-4
    DCH4 = -8.74113*10**-7
    ECH4 = 6.78119*10**-10
    F_CH4 = -2.75382*10**-13
    GCH4 = 4.58066*10**-17

    AH2O = 33.17438
    BH2O = -3.24633*10**-3
    CH2O = 1.74365*10**-5
    DH2O = -5.97957*10**-9
    EH2O = 0
    F_H2O = 0
    GH2O = 0

    ACO = 28.50457
    BCO = 1.02017*10**-2
    CCO = -6.15947*10**-5
    DCO = 1.61354*10**-7
    ECO = -1.78138*10**-10
    F_CO = 9.02011*10**-14
    GCO = -1.73591*10**-17

    ACO2 = 23.50610
    BCO2 = 3.80655*10**-2
    CCO2 = 7.40233*10**-5
    DCO2 = -2.22713*10**-7
    ECO2 = 2.34374*10**-10
    F_CO2 = -1.14647*10**-13
    GCO2 = 2.16814*10**-17

    # Variation d'enthalpie standard (298 K)

    DeltaH01 = 206*1000  # Variation enthalpie réaction 1  (J/mol)
    DeltaH02 = -41*1000  # Variation enthalpie réaction 2  (J/mol)
    DeltaH03 = 165*1000  # Variation enthalpie réaction 3  (J/mol)

    # 1er réaction
    Delta_alpha1 = 3*AH2+ACO-AH2O-ACH4
    Delta_Beta1 = 3*BH2+BCO-BH2O-BCH4
    Delta_gamma1 = 3*CH2+CCO-CH2O-CCH4
    Delta_theta1 = 3*DH2+DCO-DH2O-DCH4
    Delta_xi1 = 3*EH2+ECO-EH2O-ECH4
    Delta_epsilon1 = 3*F_H2+F_CO-F_H2O-F_CH4
    Delta_zeta1 = 3*GH2+GCO-GH2O-GCH4

    # 2e réaction
    Delta_alpha2 = AH2+ACO2-AH2O-ACO
    Delta_Beta2 = BH2+BCO2-BH2O-BCO
    Delta_gamma2 = CH2+CCO2-CH2O-CCO
    Delta_theta2 = DH2+DCO2-DH2O-DCO
    Delta_xi2 = EH2+ECO2-EH2O-ECO
    Delta_epsilon2 = F_H2+F_CO2-F_H2O-F_CO
    Delta_zeta2 = GH2+GCO2-GH2O-GCO

    # 3e réaction
    Delta_alpha3 = 4*AH2+ACO2-2*AH2O-ACH4
    Delta_Beta3 = 4*BH2+BCO2-2*BH2O-BCH4
    Delta_gamma3 = 4*CH2+CCO2-2*CH2O-CCH4
    Delta_theta3 = 4*DH2+DCO2-2*DH2O-DCH4
    Delta_xi3 = 4*EH2+ECO2-2*EH2O-ECH4
    Delta_epsilon3 = 4*F_H2+F_CO2-2*F_H2O-F_CH4
    Delta_zeta3 = 4*GH2+GCO2-2*GH2O-GCH4

    DeltaHrx = np.zeros(3)

    DeltaHrx[0] = DeltaH01+Delta_alpha1*(T-TR)+(Delta_Beta1/2)*(T**2-TR**2)+(Delta_gamma1/3)*(T**3-TR**3)+(
        Delta_theta1/4)*(T**4-TR**4)+(Delta_xi1/5)*(T**5-TR**5)+(Delta_epsilon1/6)*(T**6-TR**6)+(
            Delta_zeta1/7)*(T**7-TR**7)
    DeltaHrx[1] = DeltaH02+Delta_alpha2*(T-TR)+(Delta_Beta2/2)*(T**2-TR**2)+(Delta_gamma2/3)*(T**3-TR**3)+(
        Delta_theta2/4)*(T**4-TR**4)+(Delta_xi2/5)*(T**5-TR**5)+(Delta_epsilon2/6)*(T**6-TR**6)+(Delta_zeta2/7)*(
            T**7-TR**7)
    DeltaHrx[2] = DeltaH03+Delta_alpha3*(T-TR)+(Delta_Beta3/2)*(T**2-TR**2)+(Delta_gamma3/3)*(T**3-TR**3)+(
        Delta_theta3/4)*(T**4-TR**4)+(Delta_xi3/5)*(T**5-TR**5)+(Delta_epsilon3/6)*(T**6-TR**6)+(Delta_zeta3/7)*(
            T**7-TR**7)

    return DeltaHrx


def calc_cp(T):
    """ Calcul des capacités calorifiques en fonction de la température

    Parameters
    ----------
    T : float
        Température en Kelvin

    Returns
    -------
    Cp : array (1D)
        Capacités calorifiques pour chaque substance, dans l'ordre spécifié par IndexSMR

    """
    AH2 = 19.67099
    BH2 = 6.96815*10**-2
    CH2 = -2.0009*10**-4
    DH2 = 2.89492*10**-7
    EH2 = -2.22474*10**-10
    F_H2 = 8.81465*10**-14
    GH2 = -1.42043*10**-17

    ACH4 = 44.35658
    BCH4 = -0.14623
    CCH4 = 6.00245*10**-4
    DCH4 = -8.74113*10**-7
    ECH4 = 6.78119*10**-10
    F_CH4 = -2.75382*10**-13
    GCH4 = 4.58066*10**-17

    AH2O = 33.17438
    BH2O = -3.24633*10**-3
    CH2O = 1.74365*10**-5
    DH2O = -5.97957*10**-9
    EH2O = 0
    F_H2O = 0
    GH2O = 0

    ACO = 28.50457
    BCO = 1.02017*10**-2
    CCO = -6.15947*10**-5
    DCO = 1.61354*10**-7
    ECO = -1.78138*10**-10
    F_CO = 9.02011*10**-14
    GCO = -1.73591*10**-17

    ACO2 = 23.50610
    BCO2 = 3.80655*10**-2
    CCO2 = 7.40233*10**-5
    DCO2 = -2.22713*10**-7
    ECO2 = 2.34374*10**-10
    F_CO2 = -1.14647*10**-13
    GCO2 = 2.16814*10**-17

    Cp = np.zeros(len(ix.subst))

    Cp[ix.H2O] = AH2O+BH2O*T+CH2O*T**2+DH2O*T**3+EH2O*T**4+F_H2O*T**5+GH2O*T**6
    Cp[ix.CO2] = ACO2+BCO2*T+CCO2*T**2+DCO2*T**3+ECO2*T**4+F_CO2*T**5+GCO2*T**6
    Cp[ix.H2] = AH2+(BH2*T)+(CH2*T**2)+(DH2*T**3)+(EH2*T**4)+(F_H2*T**5)+(GH2*T**6)
    Cp[ix.CO] = ACO+BCO*T+CCO*T**2+DCO*T**3+ECO*T**4+F_CO*T**5+GCO*T**6
    Cp[ix.CH4] = ACH4+BCH4*T+CCH4*T**2+DCH4*T**3+ECH4*T**4+F_CH4*T**5+GCH4*T**6
    return Cp

# =====================================================================================================================
# FONCTIONS DE SOUTIENS TECHNIQUE
# =====================================================================================================================

def savefig(name):
    for form in ['.pdf', '.svg', '.png']:
        plt.savefig(name + form)

# https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)