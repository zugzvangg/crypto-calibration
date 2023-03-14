// HestonCalibrator.cpp
//
// Created by Yiran Cui on 09/Sep/2015.
// Last revised on 18/Mar/2016.

// Copyright 2015-2016 Yiran Cui.
// The program is distributed under the terms of the GNU General Public License.

// This script contains:
// a pricer for vanilla options under Heston model,
// a calibrator of the Heston model,
// an example of using this calibrator to find the presumed optimal parameter set.

// In order to use the calibrator with Levenberg-Marquardt method, please
// install levmar, Lourakis 2004: http://users.ics.forth.gr/~lourakis/levmar

// If you find this package useful or have any comments, questions or
// suggestions, please contact me at y.cui.12@ucl.ac.uk

// If you use this implementation in your published work or internal document,
// please include this reference:
// Yiran Cui, Sebastian del Ba\~{n}o Rollin, Guido Germano,
// Full and fast calibration of the Heston stochastic volatility model, 2015,
// arXiv:1511.08718 [q-fin.CP], http://arxiv.org/abs/1511.08718

// Bibtex Entry:
// @article{Cui2015,
// title = "Full and fast calibration of the {H}eston stochastic volatility model",
// author = "Yiran Cui and Sebastian del Ba\~{n}o Rollin and Guido Germano",
// journal = "arXiv:1511.08718",
// year = 2015}

#include <vector>
#include <complex>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <ctime>
// #include <levmar.h>
#include <iomanip>

using namespace std;

// Gauss-Legendre setup: you may choose from one of the following scheme:
// number of nodes = 64 costs around 0.25 sec to opt = 1e-12 <Recommended.>
// can ensure 1e-8 accuracy for the option price. that is, pvtrapz - pvgauss <=1e-8.
static double x64[32] = {0.0243502926634244325089558,0.0729931217877990394495429,0.1214628192961205544703765,0.1696444204239928180373136,0.2174236437400070841496487,0.2646871622087674163739642,0.3113228719902109561575127,0.3572201583376681159504426,0.4022701579639916036957668,0.4463660172534640879849477,0.4894031457070529574785263,0.5312794640198945456580139,0.5718956462026340342838781,0.6111553551723932502488530,0.6489654712546573398577612,0.6852363130542332425635584,0.7198818501716108268489402,0.7528199072605318966118638,0.7839723589433414076102205,0.8132653151227975597419233,0.8406292962525803627516915,0.8659993981540928197607834,0.8893154459951141058534040,0.9105221370785028057563807,0.9295691721319395758214902,0.9464113748584028160624815,0.9610087996520537189186141,0.9733268277899109637418535,0.9833362538846259569312993,0.9910133714767443207393824,0.9963401167719552793469245,0.9993050417357721394569056};
static double w64[32] = {0.0486909570091397203833654,0.0485754674415034269347991,0.0483447622348029571697695,0.0479993885964583077281262,0.0475401657148303086622822,0.0469681828162100173253263,0.0462847965813144172959532,0.0454916279274181444797710,0.0445905581637565630601347,0.0435837245293234533768279,0.0424735151236535890073398,0.0412625632426235286101563,0.0399537411327203413866569,0.0385501531786156291289625,0.0370551285402400460404151,0.0354722132568823838106931,0.0338051618371416093915655,0.0320579283548515535854675,0.0302346570724024788679741,0.0283396726142594832275113,0.0263774697150546586716918,0.0243527025687108733381776,0.0222701738083832541592983,0.0201348231535302093723403,0.0179517157756973430850453,0.0157260304760247193219660,0.0134630478967186425980608,0.0111681394601311288185905,0.0088467598263639477230309,0.0065044579689783628561174,0.0041470332605624676352875,0.0017832807216964329472961};

// number of nodes = 96 costs around 1.1 sec to opt = 1e-12
//static double x96[48] = {0.0162767448496029695791346,0.0488129851360497311119582,0.0812974954644255589944713,0.1136958501106659209112081,0.1459737146548969419891073,0.1780968823676186027594026,0.2100313104605672036028472,0.2417431561638400123279319,0.2731988125910491414872722,0.3043649443544963530239298,0.3352085228926254226163256,0.3656968614723136350308956,0.3957976498289086032850002,0.4254789884073005453648192,0.4547094221677430086356761,0.4834579739205963597684056,0.5116941771546676735855097,0.5393881083243574362268026,0.5665104185613971684042502,0.5930323647775720806835558,0.6189258401254685703863693,0.6441634037849671067984124,0.6687183100439161539525572,0.6925645366421715613442458,0.7156768123489676262251441,0.7380306437444001328511657,0.7596023411766474987029704,0.7803690438674332176036045,0.8003087441391408172287961,0.8194003107379316755389996,0.8376235112281871214943028,0.8549590334346014554627870,0.8713885059092965028737748,0.8868945174024204160568774,0.9014606353158523413192327,0.9150714231208980742058845,0.9277124567223086909646905,0.9393703397527552169318574,0.9500327177844376357560989,0.9596882914487425393000680,0.9683268284632642121736594,0.9759391745851364664526010,0.9825172635630146774470458,0.9880541263296237994807628,0.9925439003237626245718923,0.9959818429872092906503991,0.9983643758631816777241494,0.9996895038832307668276901};
//static double w96[48] = {0.0325506144923631662419614,0.0325161187138688359872055,0.0324471637140642693640128,0.0323438225685759284287748,0.0322062047940302506686671,0.0320344562319926632181390,0.0318287588944110065347537,0.0315893307707271685580207,0.0313164255968613558127843,0.0310103325863138374232498,0.0306713761236691490142288,0.0302999154208275937940888,0.0298963441363283859843881,0.0294610899581679059704363,0.0289946141505552365426788,0.0284974110650853856455995,0.0279700076168483344398186,0.0274129627260292428234211,0.0268268667255917621980567,0.0262123407356724139134580,0.0255700360053493614987972,0.0249006332224836102883822,0.0242048417923646912822673,0.0234833990859262198422359,0.0227370696583293740013478,0.0219666444387443491947564,0.0211729398921912989876739,0.0203567971543333245952452,0.0195190811401450224100852,0.0186606796274114673851568,0.0177825023160452608376142,0.0168854798642451724504775,0.0159705629025622913806165,0.0150387210269949380058763,0.0140909417723148609158616,0.0131282295669615726370637,0.0121516046710883196351814,0.0111621020998384985912133,0.0101607705350084157575876,0.0091486712307833866325846,0.0081268769256987592173824,0.0070964707911538652691442,0.0060585455042359616833167,0.0050142027429275176924702,0.0039645543384446866737334,0.0029107318179349464084106,0.0018539607889469217323359,0.0007967920655520124294381};

// number of nodes = 128 costs around 1.5 sec to opt = 1e-12
//static double x128[64] = {0.0122236989606157641980521,0.0366637909687334933302153,0.0610819696041395681037870,0.0854636405045154986364980,0.1097942311276437466729747,0.1340591994611877851175753,0.1582440427142249339974755,0.1823343059853371824103826,0.2063155909020792171540580,0.2301735642266599864109866,0.2538939664226943208556180,0.2774626201779044028062316,0.3008654388776772026671541,0.3240884350244133751832523,0.3471177285976355084261628,0.3699395553498590266165917,0.3925402750332674427356482,0.4149063795522750154922739,0.4370245010371041629370429,0.4588814198335521954490891,0.4804640724041720258582757,0.5017595591361444642896063,0.5227551520511754784539479,0.5434383024128103634441936,0.5637966482266180839144308,0.5838180216287630895500389,0.6034904561585486242035732,0.6228021939105849107615396,0.6417416925623075571535249,0.6602976322726460521059468,0.6784589224477192593677557,0.6962147083695143323850866,0.7135543776835874133438599,0.7304675667419088064717369,0.7469441667970619811698824,0.7629743300440947227797691,0.7785484755064119668504941,0.7936572947621932902433329,0.8082917575079136601196422,0.8224431169556438424645942,0.8361029150609068471168753,0.8492629875779689691636001,0.8619154689395484605906323,0.8740527969580317986954180,0.8856677173453972174082924,0.8967532880491581843864474,0.9073028834017568139214859,0.9173101980809605370364836,0.9267692508789478433346245,0.9356743882779163757831268,0.9440202878302201821211114,0.9518019613412643862177963,0.9590147578536999280989185,0.9656543664319652686458290,0.9717168187471365809043384,0.9771984914639073871653744,0.9820961084357185360247656,0.9864067427245862088712355,0.9901278184917343833379303,0.9932571129002129353034372,0.9957927585349811868641612,0.9977332486255140198821574,0.9990774599773758950119878,0.9998248879471319144736081};
//static double w128[64] = {0.0244461801962625182113259,0.0244315690978500450548486,0.0244023556338495820932980,0.0243585572646906258532685,0.0243002001679718653234426,0.0242273192228152481200933,0.0241399579890192849977167,0.0240381686810240526375873,0.0239220121367034556724504,0.0237915577810034006387807,0.0236468835844476151436514,0.0234880760165359131530253,0.0233152299940627601224157,0.0231284488243870278792979,0.0229278441436868469204110,0.0227135358502364613097126,0.0224856520327449668718246,0.0222443288937997651046291,0.0219897106684604914341221,0.0217219495380520753752610,0.0214412055392084601371119,0.0211476464682213485370195,0.0208414477807511491135839,0.0205227924869600694322850,0.0201918710421300411806732,0.0198488812328308622199444,0.0194940280587066028230219,0.0191275236099509454865185,0.0187495869405447086509195,0.0183604439373313432212893,0.0179603271850086859401969,0.0175494758271177046487069,0.0171281354231113768306810,0.0166965578015892045890915,0.0162550009097851870516575,0.0158037286593993468589656,0.0153430107688651440859909,0.0148731226021473142523855,0.0143943450041668461768239,0.0139069641329519852442880,0.0134112712886163323144890,0.0129075627392673472204428,0.0123961395439509229688217,0.0118773073727402795758911,0.0113513763240804166932817,0.0108186607395030762476596,0.0102794790158321571332153,0.0097341534150068058635483,0.0091830098716608743344787,0.0086263777986167497049788,0.0080645898904860579729286,0.0074979819256347286876720,0.0069268925668988135634267,0.0063516631617071887872143,0.0057726375428656985893346,0.0051901618326763302050708,0.0046045842567029551182905,0.0040162549837386423131943,0.0034255260409102157743378,0.0028327514714579910952857,0.0022382884309626187436221,0.0016425030186690295387909,0.0010458126793403487793129,0.0004493809602920903763943};

typedef struct tagGLAW{
    int numgrid; // # of nodes
    double* u; // nodes
    double* w; // weights
} GLAW;
static GLAW glaw = {64, x64, w64};
//static GLAW glaw = {96, x96, w96};
//static GLAW glaw = {128, x128, w128};

complex<double> one(1.0, 0.0), zero(0.0, 0.0), two(2.0, 0.0), i(0.0, 1.0);
const double pi = 4.0*atan(1.0), lb = 0.0, ub = 200, Q = 0.5*(ub - lb), P = 0.5*(ub + lb);

// market parameters: you may change the number of observations by modifying the size of T and K
struct mktpara{
    double S;
    double r;
    double T[40];
    double K[40];
};

// integrands for Heston pricer:
struct tagMN{
    double M1;
    double N1;
    double M2;
    double N2;
};

// return integrands (real-valued) for Heston pricer
tagMN HesIntMN(double u, double a, double b, double c, double rho, double v0,
               double K, double T, double S, double r) {
    tagMN MNbas;

    double csqr = pow(c,2);
    double PQ_M = P+Q*u, PQ_N = P-Q*u;

    complex<double> imPQ_M = i*PQ_M;
    complex<double> imPQ_N = i*PQ_N;
    complex<double> _imPQ_M = i*(PQ_M-i);
    complex<double> _imPQ_N = i*(PQ_N-i);

    complex<double> h_M = pow(K, -imPQ_M)/imPQ_M;
    complex<double> h_N = pow(K, -imPQ_N)/imPQ_N;

    double x0 = log(S) + r*T;
    
    // kes = a-i*c*rho*u1;
    double tmp = c*rho;

    
    complex<double> kes_M1 = a - tmp*_imPQ_M;
    complex<double> kes_N1 = a - tmp*_imPQ_N;
    complex<double> kes_M2 = kes_M1 + tmp;
    complex<double> kes_N2 = kes_N1 + tmp;


    // m = i*u1 + pow(u1,2);
    complex<double> m_M1 = imPQ_M + one + pow(PQ_M-i, 2); // m_M1 = (PQ_M-i)*i + pow(PQ_M-i, 2);
    complex<double> m_N1 = imPQ_N + one + pow(PQ_N-i, 2); // m_N1 = (PQ_N-i)*i + pow(PQ_N-i, 2);
    complex<double> m_M2 = imPQ_M + pow(PQ_M-zero*i, 2);
    complex<double> m_N2 = imPQ_N + pow(PQ_N-zero*i, 2);

    // d = sqrt(pow(kes,2) + m*pow(c,2));
    complex<double> d_M1 = sqrt(pow(kes_M1,2) + m_M1*csqr);
    complex<double> d_N1 = sqrt(pow(kes_N1,2) + m_N1*csqr);
    complex<double> d_M2 = sqrt(pow(kes_M2,2) + m_M2*csqr);
    complex<double> d_N2 = sqrt(pow(kes_N2,2) + m_N2*csqr);


    // g = exp(-a*b*rho*T*u1*i/c);
    double tmp1 = -a*b*rho*T/c;

    

    tmp = exp(tmp1);
    complex<double> g_M2 = exp(tmp1*imPQ_M);
    complex<double> g_N2 = exp(tmp1*imPQ_N);
    complex<double> g_M1 = g_M2*tmp;
    complex<double> g_N1 = g_N2*tmp;

    

    // alp, calp, salp
    tmp = 0.5*T;
    complex<double> alpha = d_M1*tmp;
    complex<double> calp_M1 = cosh(alpha);
    complex<double> salp_M1 = sinh(alpha);

    alpha = d_N1*tmp;
    complex<double> calp_N1 = cosh(alpha);
    complex<double> salp_N1 = sinh(alpha);


    alpha = d_M2*tmp;
    complex<double> calp_M2 = cosh(alpha);
    complex<double> salp_M2 = sinh(alpha);

    alpha = d_N2*tmp;
    complex<double> calp_N2 = cosh(alpha);
    complex<double> salp_N2 = sinh(alpha);

    // A2 = d*calp + kes*salp;
    complex<double> A2_M1 = d_M1*calp_M1 + kes_M1*salp_M1;
    complex<double> A2_N1 = d_N1*calp_N1 + kes_N1*salp_N1;
    complex<double> A2_M2 = d_M2*calp_M2 + kes_M2*salp_M2;
    complex<double> A2_N2 = d_N2*calp_N2 + kes_N2*salp_N2;

    // A1 = m*salp;
    complex<double> A1_M1 = m_M1*salp_M1;
    complex<double> A1_N1 = m_N1*salp_N1;
    complex<double> A1_M2 = m_M2*salp_M2;
    complex<double> A1_N2 = m_N2*salp_N2;

    // A = A1/A2;
    complex<double> A_M1 = A1_M1/A2_M1;
    complex<double> A_N1 = A1_N1/A2_N1;
    complex<double> A_M2 = A1_M2/A2_M2;
    complex<double> A_N2 = A1_N2/A2_N2;

    // characteristic function: y1 = exp(i*x0*u1) * exp(-v0*A) * g * exp(2*a*b/pow(c,2)*D)
    tmp = 2*a*b/csqr;
    double halft = 0.5*T;
    complex<double> D_M1 = log(d_M1) + (a - d_M1)*halft - log((d_M1 + kes_M1)*0.5 + (d_M1 - kes_M1)*0.5*exp(-d_M1*T));
    complex<double> D_M2 = log(d_M2) + (a - d_M2)*halft - log((d_M2 + kes_M2)*0.5 + (d_M1 - kes_M2)*0.5*exp(-d_M2*T));
    complex<double> D_N1 = log(d_N1) + (a - d_N1)*halft - log((d_N1 + kes_N1)*0.5 + (d_N1 - kes_N1)*0.5*exp(-d_N1*T));
    complex<double> D_N2 = log(d_N2) + (a - d_N2)*halft - log((d_N2 + kes_N2)*0.5 + (d_N2 - kes_N2)*0.5*exp(-d_N2*T));
    
    MNbas.M1 = real(h_M*exp(x0*_imPQ_M - v0*A_M1 + tmp * D_M1) * g_M1);
    MNbas.N1 = real(h_N*exp(x0*_imPQ_N - v0*A_N1 + tmp * D_N1) * g_N1);
    MNbas.M2 = real(h_M*exp(x0*imPQ_M - v0*A_M2 + tmp * D_M2) * g_M2);
    MNbas.N2 = real(h_N*exp(x0*imPQ_N - v0*A_N2 + tmp * D_N2) * g_N2);
    // cout << MNbas.M1 << " " << MNbas.N1 << " " << MNbas.M2 << " " << MNbas.N2 << " " << endl;
    // cout << salp_N1;
    return MNbas;
}

// Heston pricer: (parameter, observation, dim_p, dim_x, arguments)
void fHes(double *p, double *x, int m, int n, void *data)
{
    int l;

    // retrieve market parameters
    struct mktpara *dptr;
    dptr=(struct mktpara *)data;
    double S = dptr->S;
    double r = dptr->r;

    // retrieve model parameters
    double a = p[0];
    double b = p[1];
    double c = p[2];
    double rho = p[3];
    double v0 = p[4];

    // numerical integral settings
    int NumGrids = glaw.numgrid;
    // cout << NumGrids << "=======";
    // cout << NumGrids;
    NumGrids = (NumGrids+1)>>1;
    double *u = glaw.u;
    double *w = glaw.w;
    // cout << NumGrids;

    for (l=0; l<n; ++l) {
        double K = dptr->K[l];
        double T = dptr->T[l];
        double disc = exp(-r*T);
        double tmp = 0.5*(S-K*disc);
        disc = disc/pi;
        double Y1 = 0.0, Y2 = 0.0;
        // cout << NumGrids << "======="; 
        for (int j=0; j< NumGrids; j++) {

            tagMN MN = HesIntMN(u[j],a, b, c, rho, v0, K, T, S, r);
            
            double M1 = MN.M1;
            double N1 = MN.N1;
            double M2 = MN.M2;
            double N2 = MN.N2;
            // cout << M1 << N1 << M2 << N2 << endl;
            Y1 += w[j]*(M1+N1);
            Y2 += w[j]*(M2+N2);
        }

        double Qv1 = Q*Y1;
        double Qv2 = Q*Y2;
        double pv = tmp + disc*(Qv1-K*Qv2);
        x[l] = pv;
        
    }
    for (int j=0; j< 40; j++) {
            cout << x[j] << endl;
        }
}

// integrands for Jacobian
struct tagMNJac{
    double pa1s;
    double pa2s;

    double pb1s;
    double pb2s;

    double pc1s;
    double pc2s;

    double prho1s;
    double prho2s;

    double pv01s;
    double pv02s;
};


// return integrands (real-valued) for Jacobian
tagMNJac HesIntJac(double u, double a, double b, double c, double rho,
    double v0, double K, double T, double S, double r)
{
    tagMNJac Jacobian;

    double PQ_M = P+Q*u, PQ_N = P-Q*u;

    complex<double> imPQ_M = i*PQ_M;
    complex<double> imPQ_N = i*PQ_N;
    complex<double> _imPQ_M = i*(PQ_M-i);
    complex<double> _imPQ_N = i*(PQ_N-i);

    complex<double> h_M = pow(K,-imPQ_M)/imPQ_M;
    complex<double> h_N = pow(K,-imPQ_N)/imPQ_N;

    double x0 = log(S) + r*T;
    double tmp = c*rho;
    complex<double> kes_M1 = a - tmp*_imPQ_M;
    complex<double> kes_M2 = kes_M1 + tmp;
    complex<double> kes_N1 = a - tmp*_imPQ_N;
    complex<double> kes_N2 = kes_N1 + tmp;

    // m = i*u1 + pow(u1,2);
    complex<double> _msqr = pow(PQ_M-i, 2);
    complex<double> _nsqr = pow(PQ_N-i, 2);
    complex<double> msqr = pow(PQ_M-zero*i, 2);
    complex<double> nsqr = pow(PQ_N-zero*i, 2);

    complex<double> m_M1 = imPQ_M + one + _msqr; //    m_M1 = (PQ_M - i)*i + pow(PQ_M-i, 2);
    complex<double> m_N1 = imPQ_N + one + _nsqr; //    m_N1 = (PQ_N - i)*i + pow(PQ_N-i, 2);
    complex<double> m_M2 = imPQ_M + msqr;
    complex<double> m_N2 = imPQ_N + nsqr;

    // d = sqrt(pow(kes,2) + m*pow(c,2));
    double csqr = pow(c,2);
    complex<double> d_M1 = sqrt(pow(kes_M1,2) + m_M1*csqr);
    complex<double> d_N1 = sqrt(pow(kes_N1,2) + m_N1*csqr);
    complex<double> d_M2 = sqrt(pow(kes_M2,2) + m_M2*csqr);
    complex<double> d_N2 = sqrt(pow(kes_N2,2) + m_N2*csqr);

    // g = exp(-a*b*rho*T*u1*i/c);
    double abrt = a*b*rho*T;
    double tmp1 = -abrt/c;
    double tmp2 = exp(tmp1);

    complex<double> g_M2 = exp(tmp1*imPQ_M);
    complex<double> g_N2 = exp(tmp1*imPQ_N);
    complex<double> g_M1 = g_M2*tmp2;
    complex<double> g_N1 = g_N2*tmp2;

    // alp, calp, salp
    double halft = 0.5*T;
    complex<double> alpha = d_M1*halft;
    complex<double> calp_M1 = cosh(alpha);
    complex<double> salp_M1 = sinh(alpha);

    alpha = d_N1*halft;
    complex<double> calp_N1 = cosh(alpha);
    complex<double> salp_N1 = sinh(alpha);

    alpha = d_M2*halft;
    complex<double> calp_M2 = cosh(alpha);
    complex<double> salp_M2 = sinh(alpha);

    alpha = d_N2*halft;
    complex<double> calp_N2 = cosh(alpha);
    complex<double> salp_N2 = sinh(alpha);

    // A2 = d*calp + kes*salp;
    complex<double> A2_M1 = d_M1*calp_M1 + kes_M1*salp_M1;
    complex<double> A2_N1 = d_N1*calp_N1 + kes_N1*salp_N1;
    complex<double> A2_M2 = d_M2*calp_M2 + kes_M2*salp_M2;
    complex<double> A2_N2 = d_N2*calp_N2 + kes_N2*salp_N2;

    // A1 = m*salp;
    complex<double> A1_M1 = m_M1*salp_M1;
    complex<double> A1_N1 = m_N1*salp_N1;
    complex<double> A1_M2 = m_M2*salp_M2;
    complex<double> A1_N2 = m_N2*salp_N2;

    // A = A1/A2;
    complex<double> A_M1 = A1_M1/A2_M1;
    complex<double> A_N1 = A1_N1/A2_N1;
    complex<double> A_M2 = A1_M2/A2_M2;
    complex<double> A_N2 = A1_N2/A2_N2;

    // B = d*exp(a*T/2)/A2;
    tmp = exp(a*halft); // exp(a*T/2)
    complex<double> B_M1 = d_M1*tmp/A2_M1;
    complex<double> B_N1 = d_N1*tmp/A2_N1;
    complex<double> B_M2 = d_M2*tmp/A2_M2;
    complex<double> B_N2 = d_N2*tmp/A2_N2;

    // characteristic function: y1 = exp(i*x0*u1) * exp(-v0*A) * g * exp(2*a*b/pow(c,2)*D)
    double tmp3 = 2*a*b/csqr;
    complex<double> D_M1 = log(d_M1) + (a - d_M1)*halft - log((d_M1 + kes_M1)*0.5 + (d_M1 - kes_M1)*0.5*exp(-d_M1*T));
    complex<double> D_M2 = log(d_M2) + (a - d_M2)*halft - log((d_M2 + kes_M2)*0.5 + (d_M1 - kes_M2)*0.5*exp(-d_M2*T));
    complex<double> D_N1 = log(d_N1) + (a - d_N1)*halft - log((d_N1 + kes_N1)*0.5 + (d_N1 - kes_N1)*0.5*exp(-d_N1*T));
    complex<double> D_N2 = log(d_N2) + (a - d_N2)*halft - log((d_N2 + kes_N2)*0.5 + (d_N2 - kes_N2)*0.5*exp(-d_N2*T));
    
    
    complex<double> y1M1 = exp(x0*_imPQ_M-v0*A_M1 + tmp3*D_M1) * g_M1;
    complex<double> y1N1 = exp(x0*_imPQ_N-v0*A_N1 + tmp3*D_N1) * g_N1;
    complex<double> y1M2 = exp(x0*imPQ_M-v0*A_M2 + tmp3*D_M2) * g_M2;
    complex<double> y1N2 = exp(x0*imPQ_N-v0*A_N2 + tmp3*D_N2) * g_N2;


    // H = kes*calp + d*salp;
    complex<double> H_M1 = kes_M1*calp_M1 + d_M1*salp_M1;
    complex<double> H_M2 = kes_M2*calp_M2 + d_M2*salp_M2;
    complex<double> H_N1 = kes_N1*calp_N1 + d_N1*salp_N1;
    complex<double> H_N2 = kes_N2*calp_N2 + d_N2*salp_N2;

    // lnB = log(B);
    complex<double> lnB_M1 = D_M1; 
    complex<double> lnB_M2 = D_M2; 
    complex<double> lnB_N1 = D_N1; 
    complex<double> lnB_N2 = D_N2; 

    // partial b: y3 = y1*(2*a*lnB/pow(c,2)-a*rho*T*u1*i/c);
    double tmp4 = tmp3/b;
    double tmp5 = tmp1/b;

    complex<double> y3M1 = tmp4*lnB_M1 + tmp5*_imPQ_M;
    complex<double> y3M2 = tmp4*lnB_M2 + tmp5*imPQ_M;
    complex<double> y3N1 = tmp4*lnB_N1 + tmp5*_imPQ_N;
    complex<double> y3N2 = tmp4*lnB_N2 + tmp5*imPQ_N;

    // partial rho:
    tmp1 = tmp1/rho;//-a*b*T/c;

    // for M1
    complex<double> ctmp = c*_imPQ_M/d_M1;
    complex<double> pd_prho_M1 = -kes_M1*ctmp;
    complex<double> pA1_prho_M1 = m_M1*calp_M1*halft*pd_prho_M1;
    complex<double> pA2_prho_M1 = -ctmp* H_M1*(one+kes_M1*halft);
    complex<double> pA_prho_M1 = (pA1_prho_M1 - A_M1*pA2_prho_M1)/A2_M1;
    ctmp = pd_prho_M1 - pA2_prho_M1*d_M1/A2_M1;
    complex<double> pB_prho_M1 = tmp/A2_M1*ctmp;
    complex<double> y4M1 = -v0*pA_prho_M1 + tmp3* ctmp/d_M1 + tmp1*_imPQ_M;

    // for N1
    ctmp = c*_imPQ_N/d_N1;
    complex<double> pd_prho_N1 = -kes_N1*ctmp;
    complex<double> pA1_prho_N1 = m_N1*calp_N1*halft*pd_prho_N1;
    complex<double> pA2_prho_N1 = -ctmp*H_N1*(one+kes_N1*halft);
    complex<double> pA_prho_N1 = (pA1_prho_N1 - A_N1*pA2_prho_N1)/A2_N1;
    ctmp = pd_prho_N1 - pA2_prho_N1*d_N1/A2_N1;
    complex<double> pB_prho_N1 = tmp/A2_N1*ctmp;
    complex<double> y4N1 = -v0*pA_prho_N1 + tmp3* ctmp/d_N1 + tmp1*_imPQ_N;

    // for M2
    ctmp =c*imPQ_M/d_M2;
    complex<double> pd_prho_M2 = -kes_M2*ctmp;
    complex<double> pA1_prho_M2 = m_M2*calp_M2*halft*pd_prho_M2;
    complex<double> pA2_prho_M2 = -ctmp*H_M2*(one+kes_M2*halft)/d_M2;
    complex<double> pA_prho_M2 = (pA1_prho_M2 - A_M2*pA2_prho_M2)/A2_M2;
    ctmp = pd_prho_M2 - pA2_prho_M2*d_M2/A2_M2;
    complex<double> pB_prho_M2 = tmp/A2_M2*ctmp;
    complex<double> y4M2 = -v0*pA_prho_M2 + tmp3* ctmp/d_M2 + tmp1*imPQ_M;

    // for N2
    ctmp = c*imPQ_N/d_N2;
    complex<double> pd_prho_N2 = -kes_N2*ctmp;
    complex<double> pA1_prho_N2 = m_N2*calp_N2*halft*pd_prho_N2;
    complex<double> pA2_prho_N2 = -ctmp*H_N2*(one+kes_N2*halft);
    complex<double> pA_prho_N2 = (pA1_prho_N2 - A_N2*pA2_prho_N2)/A2_N2;
    ctmp = pd_prho_N2 - pA2_prho_N2*d_N2/A2_N2;
    complex<double> pB_prho_N2 = tmp/A2_N2*ctmp;
    complex<double> y4N2 = -v0*pA_prho_N2 + tmp3*ctmp/d_N2 + tmp1*imPQ_N;

    // partial a:
    tmp1 = b*rho*T/c;
    tmp2 = tmp3/a;//2*b/csqr;
    ctmp = -one/(c*_imPQ_M);

    complex<double> pB_pa = ctmp*pB_prho_M1 + B_M1*halft;
    complex<double> y5M1 = -v0*pA_prho_M1*ctmp + tmp2*lnB_M1 + a*tmp2*pB_pa/B_M1 - tmp1*_imPQ_M;

    ctmp = -one/(c*imPQ_M);
    pB_pa = ctmp*pB_prho_M2 + B_M2*halft;
    complex<double> y5M2 = -v0*pA_prho_M2*ctmp + tmp2*lnB_M2 + a*tmp2*pB_pa/B_M2 - tmp1*imPQ_M;

    ctmp = -one/(c*_imPQ_N);
    pB_pa = ctmp*pB_prho_N1 + B_N1*halft;
    complex<double> y5N1 = -v0*pA_prho_N1*ctmp + tmp2*lnB_N1 + a*tmp2*pB_pa/B_N1 - tmp1*_imPQ_N;

    ctmp = -one/(c*imPQ_N);
    pB_pa = ctmp*pB_prho_N2 + B_N2*halft;
    complex<double> y5N2 = -v0*pA_prho_N2*ctmp + tmp2*lnB_N2 + a*tmp2*pB_pa/B_N2 - tmp1*imPQ_N;

    // partial c:
    tmp = rho/c;
    tmp1 = 4*a*b/pow(c,3);
    tmp2 = abrt/csqr;

    // M1
    complex<double> pd_pc = (tmp - one/kes_M1)*pd_prho_M1 + c*_msqr/d_M1;
    complex<double> pA1_pc = m_M1*calp_M1*halft*pd_pc;
    complex<double> pA2_pc = tmp*pA2_prho_M1 -one/_imPQ_M*(two/(T*kes_M1)+one)*pA1_prho_M1 + c*halft*A1_M1;
    complex<double> pA_pc = pA1_pc/A2_M1 - A_M1/A2_M1*pA2_pc;
    complex<double> y6M1 = -v0*pA_pc - tmp1 *lnB_M1 + tmp3/d_M1*(pd_pc - d_M1/A2_M1*pA2_pc) +
          tmp2*_imPQ_M;

    // M2
    pd_pc = (tmp - one/kes_M2)*pd_prho_M2 + c*msqr/d_M2;
    pA1_pc = m_M2*calp_M2*halft*pd_pc;
    pA2_pc = tmp*pA2_prho_M2 - one/imPQ_M*(two/(T*kes_M2)+one)*pA1_prho_M2 + c*halft*A1_M2;
    pA_pc = pA1_pc/A2_M2 - A_M2/A2_M2*pA2_pc;
    complex<double> y6M2 = -v0*pA_pc - tmp1 *lnB_M2 + tmp3/d_M2*(pd_pc - d_M2/A2_M2*pA2_pc) +
            tmp2*imPQ_M;

    // N1
    pd_pc = (tmp - one/kes_N1)*pd_prho_N1 +  c*_nsqr/d_N1;
    pA1_pc = m_N1*calp_N1*halft*pd_pc;
    pA2_pc = tmp*pA2_prho_N1 - one/(_imPQ_N)*(two/(T*kes_N1)+one)*pA1_prho_N1 + c*halft*A1_N1;
    pA_pc = pA1_pc/A2_N1 - A_N1/A2_N1*pA2_pc;
    complex<double> y6N1 = -v0*pA_pc - tmp1 *lnB_N1 + tmp3/d_N1*(pd_pc - d_N1/A2_N1*pA2_pc) + tmp2*_imPQ_N;

    // N2
    pd_pc = (tmp - one/kes_N2)*pd_prho_N2 +  c*nsqr/d_N2;
    pA1_pc = m_N2*calp_N2*halft*pd_pc;
    pA2_pc = tmp*pA2_prho_N2 - one/(imPQ_N)*(two/(T*kes_N2)+one)*pA1_prho_N2 + c*halft*A1_N2;
    pA_pc = pA1_pc/A2_N2 - A_N2/A2_N2*pA2_pc;
    complex<double> y6N2 = -v0*pA_pc - tmp1 *lnB_N2 + tmp3/d_N2*(pd_pc - d_N2/A2_N2*pA2_pc) + tmp2*imPQ_N;

    complex<double> hM1 = h_M*y1M1;
    complex<double> hM2 = h_M*y1M2;
    complex<double> hN1 = h_N*y1N1;
    complex<double> hN2 = h_N*y1N2;

    Jacobian.pa1s = real(hM1*y5M1 + hN1*y5N1);
    Jacobian.pa2s = real(hM2*y5M2 + hN2*y5N2);

    Jacobian.pb1s = real(hM1*y3M1 + hN1*y3N1);
    Jacobian.pb2s = real(hM2*y3M2 + hN2*y3N2);

    Jacobian.pc1s = real(hM1*y6M1 + hN1*y6N1);
    Jacobian.pc2s = real(hM2*y6M2 + hN2*y6N2);

    Jacobian.prho1s = real(hM1*y4M1 + hN1*y4N1);
    Jacobian.prho2s = real(hM2*y4M2 + hN2*y4N2);

    Jacobian.pv01s = real(-hM1*A_M1 - hN1*A_N1);
    Jacobian.pv02s = real(-hM2*A_M2 - hN2*A_N2);// partial v0: y2 = -A*y1;
    // cout << Jacobian.pa1s;
    return Jacobian;
}

// Jacobian (parameter, observation, dim_p, dim_x, arguments)
void JacHes(double *p, int m, int n, void *data) {

    int l, k;
    double jac[n+10000];
    // retrieve market parameters
    struct mktpara *dptr;
    dptr=(struct mktpara *)data;
    double S = dptr->S;
    double r = dptr->r;

    // retrieve model parameters
    double a = p[0];
    double b = p[1];
    double c = p[2];
    double rho = p[3];
    double v0 = p[4];

    // numerical integration settings
    int NumGrids = glaw.numgrid;
    NumGrids = (NumGrids+1)>>1;
    double *u = glaw.u;
    double *w = glaw.w;
    for (l=k=0; l<n; ++l) {
        double K = dptr->K[l];
        double T = dptr->T[l];
        double discpi = exp(-r*T)/pi;
        double pa1 = 0.0, pa2 = 0.0, pb1 = 0.0, pb2 = 0.0, pc1 = 0.0, pc2 = 0.0, prho1 = 0.0, prho2 = 0.0, pv01 = 0.0, pv02 = 0.0;

        // integrate
        for (int j=0; j< NumGrids; j++) {
            tagMNJac jacint = HesIntJac( u[j],a, b, c, rho, v0, K, T, S, r );
            // cout << jacint.pa1s;
            pa1 += w[j]*jacint.pa1s;
            pa2 += w[j]*jacint.pa2s;

            pb1 += w[j]*jacint.pb1s;
            pb2 += w[j]*jacint.pb2s;

            pc1 += w[j]*jacint.pc1s;
            pc2 += w[j]*jacint.pc2s;

            prho1 += w[j]*jacint.prho1s;
            prho2 += w[j]*jacint.prho2s;

            pv01 += w[j]*jacint.pv01s;
            pv02 += w[j]*jacint.pv02s;
        }
        cout << pa1 << "  "<< pa2 << "  " << pb1 << "  " << pb2 << "  " 
        << pc1 << "  " << pc2 << "  " << prho1 << "  " << prho2 << "  " << pv01 << "  " << pv02 << endl;
        cout << " =======";

        double Qv1 = Q*pa1;
        double Qv2 = Q*pa2;
        // cout << discpi*(Qv1-K*Qv2);
        jac[k++] = discpi*(Qv1-K*Qv2);

        Qv1 = Q*pb1;
        Qv2 = Q*pb2;
        jac[k++] = discpi*(Qv1-K*Qv2);

        Qv1 = Q*pc1;
        Qv2 = Q*pc2;
        jac[k++] = discpi*(Qv1-K*Qv2);

        Qv1 = Q*prho1;
        Qv2 = Q*prho2;
        jac[k++] = discpi*(Qv1-K*Qv2);

        Qv1 = Q*pv01;
        Qv2 = Q*pv02;
        jac[k++] = discpi*(Qv1-K*Qv2);
    }
    cout << jac[1];
}

int main() {

    int m = 5;  // # of parameters
    int n = 40; // # of observations (consistent with the struct mktpara)

    struct mktpara market;

    // array of strikes
    double karr[] = {
        0.9371, 0.8603, 0.8112, 0.7760, 0.7470, 0.7216, 0.6699, 0.6137,
        0.9956, 0.9868, 0.9728, 0.9588, 0.9464, 0.9358, 0.9175, 0.9025,
        1.0427, 1.0463, 1.0499, 1.0530, 1.0562, 1.0593, 1.0663, 1.0766,
        1.2287, 1.2399, 1.2485, 1.2659, 1.2646, 1.2715, 1.2859, 1.3046,
        1.3939, 1.4102, 1.4291, 1.4456, 1.4603, 1.4736, 1.5005, 1.5328};

    // array of expiries
    double tarr[] = {0.119047619047619, 0.238095238095238,	0.357142857142857, 0.476190476190476,	0.595238095238095, 0.714285714285714, 1.07142857142857, 1.42857142857143,
        0.119047619047619	,0.238095238095238, 0.357142857142857, 0.476190476190476, 0.595238095238095, 0.714285714285714	,1.07142857142857, 1.42857142857143	,
        0.119047619047619, 	0.238095238095238,	0.357142857142857,	0.476190476190476,	0.595238095238095,	0.714285714285714,	1.07142857142857,	1.42857142857143,
        0.119047619047619,	0.238095238095238,	0.357142857142857,	0.476190476190476	,0.595238095238095,	0.714285714285714,	1.07142857142857,	1.42857142857143,
        0.119047619047619,	0.238095238095238	,0.357142857142857,	0.476190476190476,	0.595238095238095,	0.714285714285714,	1.07142857142857,	1.42857142857143};

    // strikes and expiries
    for (int j=0; j<n; ++j) {
        market.K[j] = karr[j];
        market.T[j] = tarr[j];
    }

    // spot and interest rate
    market.S = 1.0;
    market.r = 0.02;

    // you may set up your optimal model parameters here:
    // set optimal model parameters  | Corresponding model paramater |  Meaning
    double a = 3.0;               // kappa                           |  mean reversion rate
    double b = 0.10;              // v_infinity                      |  long term variance
    double c = 0.25;              // sigma                           |  variance of volatility
    double rho = -0.8;            // rho                             |  correlation between spot and volatility
    double v0 = 0.08;             // v0                              |  initial variance

    double pstar[5];
    pstar[0] = a; pstar[1] = b; pstar[2] = c; pstar[3] = rho; pstar[4] = v0;

    // compute the market observatoins with pstar
    double x[40];
    // fHes(pstar, x, m, n, (void *) &market);
    // for(int j =0; j<32; j++){
    //     HesIntMN(x64[j], a, b, c, rho, v0, karr[0], tarr[0], market.S, market.r);
    // }
    JacHes(pstar, m, n, (void *) &market);
    // for(int j =0; j<32; j++){
    //     tagMNJac jacint = HesIntJac(x64[j], a, b, c, rho, v0, karr[0], tarr[0], market.S, market.r);
    // }
    
    
    

    // >>> Enter calibrating routine >>>
    // double start_s = clock();

    // algorithm parameters
    // double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
    // opts[0]=LM_INIT_MU;
    // // stopping thresholds for
    // opts[1]=1E-10;       // ||J^T e||_inf
    // opts[2]=1E-10;       // ||Dp||_2
    // opts[3]=1E-10;       // ||e||_2
    // opts[4]= LM_DIFF_DELTA; // finite difference if used

    // you may set up your initial point here:
    // double p[5];
    // p[0] = 1.2000;
    // p[1] = 0.20000;
    // p[2] = 0.3000;
    // p[3] = -0.6000;
    // p[4] = 0.2000;

    // cout << "\r-------- -------- -------- Heston Model Calibrator -------- -------- --------"<<endl;
    // cout << "Parameters:" << "\t         kappa"<<"\t     vinf"<< "\t       vov"<< "\t      rho" << "\t     v0"<<endl;
    // cout << "\r Initial point:" << "\t"  << scientific << setprecision(8) << p[0]<< "\t" << p[1]<< "\t"<< p[2]<< "\t"<< p[3]<< "\t"<< p[4] << endl;
    // Calibrate using analytical gradient
    // dlevmar_der(fHes, JacHes, p, x, m, n, 100, opts, info, NULL, NULL, (void *) &market);

    // double stop_s = clock();

    // cout << "Optimum found:" << scientific << setprecision(8) << "\t"<< p[0]<< "\t" << p[1]<< "\t"<< p[2]<< "\t"<< p[3]<< "\t"<< p[4] << endl;
    // cout << "Real optimum:" << "\t" << pstar[0]<<"\t"<< pstar[1]<< "\t"<< pstar[2]<< "\t"<< pstar[3]<< "\t"<< pstar[4] << endl;

    // if (int(info[6]) == 6) {
    //     cout << "\r Solved: stopped by small ||e||_2 = "<< info[1] << " < " << opts[3]<< endl;
    // } else if (int(info[6]) == 1) {
    //     cout << "\r Solved: stopped by small gradient J^T e = " << info[2] << " < " << opts[1]<< endl;
    // } else if (int(info[6]) == 2) {
    //     cout << "\r Solved: stopped by small change Dp = " << info[3] << " < " << opts[2]<< endl;
    // } else if (int(info[6]) == 3) {
    //     cout << "\r Unsolved: stopped by itmax " << endl;
    // } else if (int(info[6]) == 4) {
    //     cout << "\r Unsolved: singular matrix. Restart from current p with increased mu"<< endl;
    // } else if (int(info[6]) == 5) {
    //     cout << "\r Unsolved: no further error reduction is possible. Restart with increased mu"<< endl;
    // } else if (int(info[6]) == 7) {
    //     cout << "\r Unsolved: stopped by invalid values, user error"<< endl;
    // }

    // cout << "\r-------- -------- -------- Computational cost -------- -------- --------"<<endl;
    // cout << "\r          Time cost: "<< double(stop_s - start_s) /CLOCKS_PER_SEC << " seconds "<<endl;
    // cout << "         Iterations: " << int(info[5]) << endl;
    // cout << "         pv  Evalue: " << int(info[7]) << endl;
    // cout << "         Jac Evalue: "<< int(info[8]) << endl;
    // cout << "# of lin sys solved: " << int(info[9])<< endl; //The attempts to reduce error
    // cout << "\r-------- -------- -------- Residuals -------- -------- --------"<<endl;
    // cout << " \r            ||e0||_2: " << info[0] << endl;
    // cout << "           ||e*||_2: " << info[1]<<endl;
    // cout << "          ||J'e||_inf: " << info[2]<<endl;
    // cout << "           ||Dp||_2: " << info[3]<<endl;

    return 0;
} // The End
