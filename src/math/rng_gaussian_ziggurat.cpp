/*

   calin/math/rng_gaussian_ziggurat.cpp -- Stephen Fegan -- 2022-09-25

   Constants for Gaussian Ziggurat - generated by python notebook
   ~/Google Drive/calin/Ziggurat - Gaussian.ipynb

   Copyright 2022, Stephen Fegan <sfegan@llr.in2p3.fr>
   Laboratoire Leprince-Ringuet, CNRS/IN2P3, Ecole Polytechnique, Institut Polytechnique de Paris

   This file is part of "calin"

   "calin" is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License version 2 or
   later, as published by the Free Software Foundation.

   "calin" is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

*/

#include <stdint.h>
#include <math/rng.hpp>

double calin::math::rng::gaussian_ziggurat::xi[257] = {
  0.0,
  0.215241895984881699325976136998,
  0.286174591792072510002201653696,
  0.335737519214425235638195399115,
  0.375121332878380591495093443902,
  0.408389134611991145290558016392,
  0.437518402207871681933515025153,
  0.463634336790882217507922976774,
  0.487443966139236039301073245078,
  0.50942332960209181446982329905,
  0.529909720661558116786810165392,
  0.549151702327165120668100504054,
  0.567338257053818748196811565987,
  0.58461676610637932144158760128,
  0.60110461775599262153390068287,
  0.616896990007751449983468424569,
  0.632072236386061170945000136827,
  0.646695714894993817513389454391,
  0.660822574244419738417074112835,
  0.674499822837293822822291441943,
  0.687767892795788534294858623405,
  0.700661841106815072627797458747,
  0.713212285190975958395437234216,
  0.725446140909999639160421404672,
  0.737387211434295591278302895312,
  0.749056662017815292302576863541,
  0.760473406430108029348112105514,
  0.771654424224568084749572873226,
  0.782615023307233120893558043738,
  0.793369058840623296211094246667,
  0.80392911698997122040753951803,
  0.81430667013521523039269489999,
  0.824512208752292130518310689444,
  0.834555354086382178924726895187,
  0.844444954909153918889582440725,
  0.854189171008163807454180162982,
  0.863795545553308854813178505387,
  0.873271068088860754125762716214,
  0.882622229585165554772936671615,
  0.891855070732941566850586789353,
  0.90097522446122183374654785638,
  0.909987953496718494483529567684,
  0.918898183649590612180034442449,
  0.927710533402000123870677193347,
  0.936429340286575141234349646198,
  0.945058684468165463037907528947,
  0.953602409881086036147944393705,
  0.962064143223040583869351754878,
  0.970447311064224450680967868246,
  0.9787551552942246038808242096,
  0.986990747099062472368807733561,
  0.995156999635090923837912834285,
  1.0032566795446729770638384769,
  1.01129241743999573953048904119,
  1.01926671746548424496253746752,
  1.02718196603564577235063988793,
  1.03504043983344087588786274603,
  1.04284431314414897093998022902,
  1.05059566459092990151427932907,
  1.05829648333067508451396689623,
  1.06594867476212250175428402653,
  1.07355406579243628829268650046,
  1.08111440970340383807753813675,
  1.08863139065397982140399321344,
  1.09610662785202143709493836541,
  1.10354167942463971835070954684,
  1.11093804601357572141756077925,
  1.11829717411934498154669330934,
  1.12562045921553339122747559385,
  1.13290924865253375311589871589,
  1.14016484436815124266444893793,
  1.14738850542084905845875724358,
  1.1545814503599277407406692525,
  1.16174485944561144240767865873,
  1.16887987673083313838407680221,
  1.17598761201545209843794068703,
  1.18306914268268676102992149751,
  1.19012551542669206932047979568,
  1.19715774787944155514995116994,
  1.20416683014438151297258540922,
  1.21115372624369918303592770168,
  1.21811937548548165603649292623,
  1.22506469375653078709685939198,
  1.23199057474613609135459618398,
  1.23889789110568737449397567271,
  1.24578749554862729459666205536,
  1.25266022189489722735447386561,
  1.259516886063714228190559472,
  1.26635828701822945377531384885,
  1.27318520766535636476406032372,
  1.27999841571381792479730890791,
  1.28679866449324364631666089651,
  1.29358669373694775394560140636,
  1.30036323033083719035192943965,
  1.3071289890307311101781311955,
  1.31388467315022048985473680895,
  1.32063097522105626431636328188,
  1.32736857762792585364443208022,
  1.33409815321936004566795918172,
  1.34082036589640403879774051436,
  1.34753587118058719822630319191,
  1.35424531676263499500784376607,
  1.36094934303328301139652884428,
  1.36764858359747620266273387874,
  1.37434366577316625980933040502,
  1.38103521107585542655702388605,
  1.38772383568997604281645761016,
  1.39441015092814101391081820904,
  1.40109476367925097737246802739,
  1.40777827684639882989072954411,
  1.41446128977547119072909197577,
  1.42114439867530904867832884004,
  1.42782819703025602804692358026,
  1.43451327600589220037319782135,
  1.44120022484872396989648390546,
  1.44788963128057610033896060321,
  1.45458208188841027520247361489,
  1.4612781625102755581416345729,
  1.46797845861807962496250555102,
  1.47468355569785557062886519653,
  1.48139403962818736390226302894,
  1.48811049705744755301310327122,
  1.49483351578049355503589015932,
  1.50156368511546373868817327368,
  1.50830159628131149617142518378,
  1.51504784277671456694719425573,
  1.52180302076099800867922849863,
  1.52856772943771240262819946144,
  1.53534257144151413808270713021,
  1.54212815322900195931807594565,
  1.54892508547417340637527413333,
  1.55573398346917637599103861956,
  1.56255546753104482093062725423,
  1.56939016341512365604283237095,
  1.57623870273590632087665800297,
  1.58310172339602924752110705806,
  1.58997987002419079746111174786,
  1.59687379442278817556308797882,
  1.60378415602609453039306680522,
  1.61071162236983005116054524904,
  1.61765686957301553263788061976,
  1.62462058283303477835405945586,
  1.63160345693487354647153268346,
  1.63860619677554773019120537389,
  1.64562951790478234609999743507,
  1.65267414708305594497707666628,
  1.65974082285818255238478974741,
  1.66683029616166551228071720036,
  1.67394333092612499923191172923,
  1.68108070472517387190961730339,
  1.68824320943719538909369501826,
  1.69543165193456156829958808919,
  1.70264685479992315165390040705,
  1.70988965707130182024174548201,
  1.71716091501782308974165911961,
  1.72446150294804491205286239777,
  1.73179231405296315413793350581,
  1.73915426128591165726242005563,
  1.74654827828172241285361081991,
  1.75397532031767153517628649565,
  1.76143636531891028053979425159,
  1.76893241491126858902966515984,
  1.77646449552452286899612411927,
  1.78403365954944151297186407781,
  1.79164098655216259462438232213,
  1.79928758454972019934172868098,
  1.80697459135082093870342042839,
  1.81470317596628267168077234352,
  1.82247454009388583887189857162,
  1.83028991968275693375605567698,
  1.8381505865828066337649313049,
  1.84605785028518550557058095299,
  1.85401305976020190675089844118,
  1.8620176053996741186653487844,
  1.87007292107126677849633777629,
  1.87818048629299584468446799604,
  1.88634182853678282003785655331,
  1.89455852567070473204088569057,
  1.90283220855042926945278439923,
  1.9111645637712533383490161194,
  1.91955733659318811306428206346,
  1.92801233405266571032880895751,
  1.93653142827569470038070219713,
  1.94511656000867830123468628187,
  1.95376974238464677669257281182,
  1.96249306494436305282602438126,
  1.97128869793365929460635647443,
  1.98015889690047660554041669481,
  1.98910600761743812320102309122,
  1.99813247135841968039221476916,
  2.0072408305605287589132397382,
  2.01643373490620412387398857702,
  2.02571394786385424525239902595,
  2.03508435372961897141387194869,
  2.04454796521753145528262879275,
  2.05410793165065213021947566631,
  2.06376754781173211434185374908,
  2.07353026351874303464639324842,
  2.08339969399830461367079208818,
  2.09337963113879193016636158596,
  2.10347405571487730593371430404,
  2.11368715068665317778193519859,
  2.12402331568952354542071478748,
  2.1344871828460169091788366048,
  2.14508363404788898276799972957,
  2.1558178198767374691195036772,
  2.16669518035430854235313714212,
  2.17772146774029300257916407915,
  2.1889027716263607428395765056,
  2.20024554661127642771216517329,
  2.21175664288416099747050070927,
  2.22344334009251061136534640972,
  2.23531338492992111074836219967,
  2.24737503294738926229795239251,
  2.25963709517378762459756653117,
  2.27210899022838186193768371737,
  2.28480080272449212738783448694,
  2.29772334890286352007979081423,
  2.31088825060137175855061435586,
  2.32430801887113250826611915705,
  2.33799614879652863543348032709,
  2.35196722737914476190253075145,
  2.36623705671729091136214812814,
  2.38082279517208555650661969132,
  2.39574311978192735616868668141,
  2.41101841390111949169034921172,
  2.42667098493714671986352985163,
  2.44272531820036422379423491922,
  2.45920837433470503567385959649,
  2.47614993967052316375621626816,
  2.49358304127104676817005329651,
  2.51154444162669434325460734458,
  2.53007523215985418771653908413,
  2.54922155032478310442267137124,
  2.56903545268184378131426292153,
  2.58957598670828664980880557451,
  2.61091051848882367193026369448,
  2.63311639363158275997630929252,
  2.65628303757674329680212430456,
  2.68051464328574510109837461143,
  2.705933656123062221333700226,
  2.73268535904401142004318251305,
  2.76094400527998620124438239249,
  2.79092117400192731899777904547,
  2.82287739682644290753411551565,
  2.85713873087322458856164526805,
  2.89412105361341218138810035621,
  2.93436686720888758995992897957,
  2.97860327988184316553697421229,
  3.02783779176959352457171458422,
  3.08352613200214325187776894762,
  3.14788928951800068545185519408,
  3.22457505204780158714401982876,
  3.32024473383982551753223298444,
  3.44927829856143127062722821383,
  3.6541528853610087716454297204,
  3.91075795952491586954962143453,
};

double calin::math::rng::gaussian_ziggurat::yi[257] = {
  1.0,
  0.977101701267671240395871902877,
  0.95987909180010638647213553683,
  0.945198953442299366337493705909,
  0.932060075959230180679446672988,
  0.91999150503934670305469193934,
  0.90872644005213053713584951301,
  0.898095921898343136073655180021,
  0.887984660755833058528578280898,
  0.878309655808917064298819305748,
  0.869008688036856712917110897167,
  0.860033621196331239099173078561,
  0.851346258458677690803928050672,
  0.842915653112203906521008471021,
  0.834716292986883159581345534631,
  0.826726833946221078452998460578,
  0.818929191603702036983822838656,
  0.811307874312655905989103435314,
  0.803849483170963949600977377325,
  0.796542330422958596548359431195,
  0.789376143566024184710479014294,
  0.782341832654802055066211608696,
  0.775431304981186743319478446398,
  0.768637315798485836478008252838,
  0.761953346836794911389454756325,
  0.755373506507095749634207059037,
  0.748892447219156494978748477602,
  0.742505296340150706880711350676,
  0.736207598126862263693175730565,
  0.729995264561475821525596509194,
  0.723864533468629809955856869535,
  0.717811932630721591015946924576,
  0.711834248878248042556796930431,
  0.705928501332753889503645907468,
  0.70009191813651123321610217299,
  0.694321916126116336554577206152,
  0.68861608300467141482002486821,
  0.682972161644994430492044843458,
  0.677388036218773132031630328063,
  0.67186171989708177426020526882,
  0.666391343908749825963766619882,
  0.660975147776662865732542520957,
  0.655611470579697006683228640829,
  0.650298743110816432653160324974,
  0.645035480820822042672541812371,
  0.639820277453056282585779979512,
  0.634651799287623311864684370747,
  0.629528779924836381640727584004,
  0.62445001554702620447897248361,
  0.619414360605834051302513352721,
  0.614420723888913570032720884788,
  0.609468064925773155450753190645,
  0.604555390697467465347208193596,
  0.599681752619124912762956498874,
  0.594846243767986980275379581443,
  0.590047996332825508494238548586,
  0.585286179263370928122835519582,
  0.580559996100790474075710709996,
  0.575868682972353254347758621765,
  0.571211506735252744960955598506,
  0.566587763256163998142355397661,
  0.56199677581452398134496175749,
  0.557437893618765622792252844843,
  0.552910490425831947587624451882,
  0.548413963255265431337961037354,
  0.543947731190025858432722779165,
  0.539511234256951674473297468879,
  0.53510393238045720164881294975,
  0.530725304403661592596544011317,
  0.526374847171684053684257483741,
  0.522052074672321477933793687353,
  0.517756517229755959080088098245,
  0.513487720747326592945928175928,
  0.509245245995747631963469878339,
  0.505028667943467912181406687163,
  0.50083757512614842653312928971,
  0.496671569052489395914524488647,
  0.492530263643868194164330014681,
  0.488413284705457654866164811658,
  0.484320269426682952519653261069,
  0.480250865909046459664847617936,
  0.476204732719505547957438615869,
  0.472181538467729829666108893458,
  0.468180961405693249169405395651,
  0.4642026890481739948349304599,
  0.460246417812842533297959998065,
  0.45631185267871617077655722894,
  0.452398706861848311585471383208,
  0.448506701507202808945304279197,
  0.444635565395739199788925298283,
  0.44078503466580382132023318163,
  0.436954852547985394401574099124,
  0.433144769112652140770657540927,
  0.429354541029441339018350737598,
  0.4255839313380218334216494213,
  0.421832709229495764235465170405,
  0.418100649837848025365765156828,
  0.414387534040890980223301386504,
  0.410693148270188054502374072516,
  0.40701728432947322499213313578,
  0.403359739221114362221750581813,
  0.399720314980197066658766786259,
  0.39609881851583224934222260077,
  0.392495061459315417294355032914,
  0.388908860018788585606542209116,
  0.385340034840077091775467710874,
  0.381788410873393461312885569108,
  0.378253817245618983313054275411,
  0.374736087137890906869194980054,
  0.371235057668239261171520028106,
  0.367750569779032322702356047675,
  0.364282468129003783574721770429,
  0.360830600989647790342169600425,
  0.357394820145780290967286636012,
  0.353974980800076611891624797323,
  0.350570941481405945018171189943,
  0.347182563956793508974948019794,
  0.343809713146850609124926379664,
  0.340452257044521701421836469288,
  0.337110066637005907870199201737,
  0.333783015830718274312500615561,
  0.330470981379163439891743986335,
  0.327173842813601334504455909397,
  0.323891482376391066103779097082,
  0.32062378495690533183472198318,
  0.317370638029913511637965630181,
  0.314131931596337104225714915015,
  0.310907558126286365578454593422,
  0.307697412504291930129463889664,
  0.304501391976649853949488366544,
  0.301319396100802935553277249447,
  0.298151326696685360250912846304,
  0.294997087799961693981676506436,
  0.291856585617095037005313342471,
  0.288729728482182750466810656498,
  0.285616426815501602630677683024,
  0.282516593083707458659916077678,
  0.279430141761637769648575428971,
  0.276356989295668114018097930962,
  0.273297054068576917586798645545,
  0.270250256365875237321557982173,
  0.267216518343561147208449299013,
  0.264195763997260821635581246906,
  0.261187919132720880588085933684,
  0.258192911337618949897618335474,
  0.255210669954661706559588120873,
  0.252241126055941931271118558204,
  0.249284212418528285179459411019,
  0.246339863501263672518634384242,
  0.243408015422750152370338544706,
  0.240488605940500428134755504078,
  0.237581574431237979299553746552,
  0.234686861872329913590874553444,
  0.231804410824338615436082411216,
  0.228934165414680255779180454933,
  0.226076071322380215679480752677,
  0.223230075763917468947413296002,
  0.220396127480151974670218530317,
  0.217574176724331156431446739693,
  0.214764175251173599194246738734,
  0.21196607630703018540245364309,
  0.20917983462112502745810306451,
  0.206405406397880743425349646375,
  0.203642749310334876188234850125,
  0.20089182249465658356226170251,
  0.198152586545775138929618342735,
  0.195425003514134290523349849909,
  0.192709036903589145103229867716,
  0.190004651670464982023773548742,
  0.187311814223800281299290402956,
  0.184630492426799280198061210479,
  0.181960655599522574550952745116,
  0.179302274522847672346823767404,
  0.176655321443735009119174590322,
  0.174019770081838769944532376055,
  0.171395595637505956692502300207,
  0.168782774801211519176269703317,
  0.16618128576448206562715073618,
  0.163591108232365715292278773215,
  0.161012223437511091441599605316,
  0.15844461415592431833229614839,
  0.155888264724479227074541739878,
  0.153343161060262844547948521826,
  0.15080929068184569423746624263,
  0.148286642732574542639836219549,
  0.145775208005994052069503721555,
  0.143274978973513431463306644558,
  0.140785949814444702456060122846,
  0.138308116448550721439934559548,
  0.135841476571253735623172714887,
  0.133386029691669133532664805494,
  0.130941777173644326655605132769,
  0.128508722279999537750691402469,
  0.126086870220185864691330036439,
  0.123676228201596555205067830299,
  0.121276805484790217220399034482,
  0.118888613442909987396360645575,
  0.116511665625610814356309442004,
  0.114145977827838359609028821163,
  0.111791568163838011701901968895,
  0.109448457146811648400260787769,
  0.107116667774683646976269569491,
  0.104796225622486906213263677134,
  0.102487158941935087424598077356,
  0.100189498768809817690278590173,
  0.0979032790388622933779394185436,
  0.0956285367130088271281738890584,
  0.0933653119126908689893965551847,
  0.0911136480663736396217164785513,
  0.0888735920682757973552654968493,
  0.0866451944505579647769562377127,
  0.0844285095703533770252891638299,
  0.0822235958132028675658029994061,
  0.0800305158146630622586731323772,
  0.0778493367020960536122838648153,
  0.0756801303589270809109427151059,
  0.0735229737139812724839391153782,
  0.0713779490588903799503404738999,
  0.0692451443970067744331979279131,
  0.0671246538277884966347671666157,
  0.0650165779712428542157030914,
  0.0629210244377581220316118631527,
  0.0608381083495398676353148831863,
  0.0587679529209337619472286373993,
  0.056710690106202903787266152376,
  0.0546664613248889217828303621547,
  0.0526354182767921839941131867706,
  0.0506177238609477654315029693632,
  0.0486135532158685252531739303297,
  0.0466230949019303696117133590936,
  0.0446465522512944488608565046959,
  0.0426841449164744372999029137841,
  0.0407361106559409331669316602131,
  0.0388027074045261164997548176785,
  0.0368842156885672883452368787172,
  0.0349809414617160853693540570187,
  0.0330932194585785224043545345375,
  0.0312214171919202488953430278963,
  0.0293659397581333157542051942948,
  0.0275272356696030839640501462301,
  0.0257058040085488981177638557824,
  0.0239022033057958826655949501297,
  0.0221170627073088660437188131161,
  0.0203510962300445204307026918331,
  0.0186051212757246457712819843289,
  0.0168800831525431684609402925449,
  0.0151770883079353269779398956148,
  0.0134974506017398801238472695625,
  0.0118427578579078889753040161234,
  0.0102149714397014714372238424176,
  0.00861658276939873193774138937097,
  0.00705087547137322698366052438995,
  0.0055224032992509972320582455537,
  0.00403797259336303082185788154245,
  0.00260907274610216295131325492458,
  0.00126028593049859756413346221554,
  0.0,
};

double calin::math::rng::gaussian_ziggurat::r = 3.6541528853610087716454297204;
double calin::math::rng::gaussian_ziggurat::r_inv = 0.273661237329758273883665255545;
double calin::math::rng::gaussian_ziggurat::v = 0.00492867323397465534736177540235;