using LinearAlgebra
using DiffEqBase
using LsqFit

abstract type AbstractInterface end
export AbstractInterface

abstract type AbstractBoundaryInterface <: AbstractInterface end
export AbstractBoundaryInterface

abstract type AbstractInternalInterface <: AbstractInterface end
export AbstractInternalInterface

abstract type AbstractReactiveInternalInterface <: AbstractInternalInterface end
export AbstractReactiveInternalInterface

struct EmptyInterface <: AbstractInterface end
export EmptyInterface

struct ReactiveInternalInterface{T,B,C,C2,N,Q<:AbstractReaction,X} <: AbstractReactiveInternalInterface
    domain1::T
    domain2::N
    reactions::Array{Q,1}
    veckinetics::X
    veckineticsinds::Array{Int64,1}
    rxnarray::B
    stoichmatrix::C
    Nrp1::C2
    Nrp2::C2
    A::Float64
    parameterindexes::Array{Int64,1}
    domaininds::Array{Int64,1}
    p::Array{Float64,1}
    reversibililty::Array{Bool,1}
end
function ReactiveInternalInterface(domain1,domain2,reactions,A)
    vectuple,vecinds,otherrxns,otherrxninds,posinds = getveckinetics(reactions)
    rxns = vcat(reactions[vecinds],reactions[otherrxninds])
    rxns = [ElementaryReaction(index=i,reactants=rxn.reactants,reactantinds=rxn.reactantinds,products=rxn.products,
        productinds=rxn.productinds,kinetics=rxn.kinetics,radicalchange=rxn.radicalchange,reversible=rxn.reversible,pairs=rxn.pairs) for (i,rxn) in enumerate(rxns)]
    rxnarray = getinterfacereactioninds(domain1,domain2,rxns)
    M,Nrp1,Nrp2 = getstoichmatrix(domain1,domain2,reactions)
    reversibility = Array{Bool,1}(getfield.(rxns,:reversible))
    return ReactiveInternalInterface(domain1,domain2,
            rxns,vectuple,posinds,rxnarray,M,Nrp1,Nrp2,A,[1,length(reactions)],
            [0,1],ones(length(rxns)),reversibility),ones(length(rxns))
end
export ReactiveInternalInterface

function getkfskrevs(ri::ReactiveInternalInterface,T1,T2,phi1,phi2,Gs1,Gs2,cstot::Array{Q,1}) where {Q}
    kfs = getkfs(ri,T1,0.0,0.0,Array{Q,1}(),ri.A,phi1)
    Kc = getKc.(ri.reactions,ri.domain1.phase,ri.domain2.phase,Ref(Gs1),Ref(Gs2),T1,phi1)
    krevs = kfs./Kc
    return kfs,krevs
end

function evaluate(ri::ReactiveInternalInterface,dydt,domains,T1,T2,phi1,phi2,Gs1,Gs2,cstot,p::W) where {W<:DiffEqBase.NullParameters}
    kfs,krevs = getkfskrevs(ri,T1,T2,phi1,phi2,Gs1,Gs2,cstot)
    addreactionratecontributions!(dydt,ri.rxnarray,cstot,kfs,krevs,ri.A)
end

function evaluate(ri::ReactiveInternalInterface,dydt,domains,T1,T2,phi1,phi2,Gs1,Gs2,cstot,p)
    kfs,krevs = getkfskrevs(ri,T1,T2,phi1,phi2,Gs1,Gs2,cstot)
    addreactionratecontributions!(dydt,ri.rxnarray,cstot,kfs.*p[ri.parameterindexes[1]:ri.parameterindexes[2]],krevs.*p[ri.parameterindexes[1]:ri.parameterindexes[2]],ri.A)
end
export evaluate

struct DiffusiveInternalInterface{T,B,N} <: AbstractInternalInterface
    domain1::T
    domain2::N
    diffusivespcnames::Array{String,1}
    diffusionarray::B
    parameterindexes::Array{Int64,1}
    A::Float64
    L::Float64
    domaininds::Array{Int64,1}
    p::Array{Float64,1}
end
function DiffusiveInternalInterface(domain1,domain2,domains,diffusivespcnames,A;L=1e-6)
    domaininds = Array{Int64,1}([0,0])
    diffusionarray = getinterfacediffusioninds(domain1,domain2,diffusivespcnames)
    for (i,domain) in enumerate(domains)
        if domain==domain1
            domaininds[1]=i
        elseif domain==domain2
            domaininds[2]=i
        end
    end
    return DiffusiveInternalInterface(domain1,domain2,diffusivespcnames,diffusionarray,[1,length(diffusivespcnames)],A,L,domaininds,ones(length(diffusivespcnames))),ones(length(diffusivespcnames))
end
export DiffusiveInternalInterface

function getdiffs(di::DiffusiveInternalInterface,T1,T2) where {Q}
    phase = di.domain1.phase
    if :solvent in fieldnames(typeof(phase)) && typeof(phase.solvent) != EmptySolvent
        mu = phase.solvent.mu(T1)
    else
        mu = 0.0
    end
    P = 1.0e8
    diffs = [x(T=T1,mu=mu,P=P) for x in getfield.(phase.species,:diffusion)[di.diffusionarray[1,:]]]
    return diffs
end

function evaluate(di::DiffusiveInternalInterface,dydt,V1,V2,T1,T2,cstot,p::W) where {W<:DiffEqBase.NullParameters}
    diffs = getdiffs(di,T1,T2)
    if isa(di.domain1,ConstantTrhoDomain)
        L = V1/di.domain1.A
    elseif isa(di.domain2,ConstantTrhoDomain)
        L = V2/di.domain2.A
    else
        L = di.L
    end
    addreactionratecontributions!(dydt,di.diffusionarray,cstot,diffs./L,diffs./L,di.A)
end

function evaluate(di::DiffusiveInternalInterface,dydt,V1,V2,T1,T2,cstot,p)
    diffs = getdiffs(di,T1,T2)
    if isa(di.domain1,ConstantTrhoDomain)
        L = V1/di.domain1.A
    elseif isa(di.domain2,ConstantTrhoDomain)
        L = V2/di.domain2.A
    else
        L = di.L
    end
    addreactionratecontributions!(dydt,di.diffusionarray,cstot,diffs./L.*p[di.parameterindexes[1]:di.parameterindexes[2]],diffs./L.*p[di.parameterindexes[1]:di.parameterindexes[2]],di.A)
end
export evaluate

struct VaporLiquidMassTransferInternalInterfaceConstantT{D1,D2,B} <: AbstractInternalInterface
    domain1::D1
    domain2::D2
    masstransferspcnames::Array{String,1}
    masstransferarray::B
    kLAs::Array{Float64,1}
    kHs::Array{Float64,1}
    parameterindexes::Array{Int64,1}
    domaininds::Array{Int64,1}
    p::Array{Float64,1}
end

function VaporLiquidMassTransferInternalInterfaceConstantT(domain1,domain2,masstransferspcnames)
    @assert isa(domain1.phase,IdealGas)
    @assert isa(domain2.phase,IdealDiluteSolution)
    T = domain2.T
    phase = domain2.phase
    masstransferarray = zeros(Int64,(6,length(masstransferspcnames)))
    kLAs = [kLA(T=T) for kLA in getfield.(phase.species,:liquidvolumetricmasstransfercoefficient)]
    kHs = [kH(T=T) for kH in getfield.(phase.species,:henrylawconstant)]
    return VaporLiquidMassTransferInternalInterfaceConstantT(domain1,domain2,masstransferspcnames,masstransferarray,kLAs,kHs,[1,length(masstransferspcnames)],[0,0],ones(length(masstransferspcnames))),ones(length(masstransferspcnames))
end
export VaporLiquidMassTransferInternalInterfaceConstantT

function getkLAkHs(vl::VaporLiquidMassTransferInternalInterfaceConstantT,T1,T2)    
    return vl.kLAs, vl.kHs
end

function evaluate(vl::VaporLiquidMassTransferInternalInterfaceConstantT,dydt,V1,V2,T1,T2,P1,P2,cstot,p::W) where {W<:DiffEqBase.NullParameters}
    kLAs, kHs = getkLAkHs(vl,T1,T2)
    @views @inbounds @fastmath evap = kLAs.*cstot[vl.masstransferarray[1,:]]*V2
    @views @inbounds @fastmath cond = kLAs./kHs.*cstot[vl.masstransferarray[4,:]]*V1
    @views @inbounds @fastmath dydt[vl.masstransferarray[1,:]] .-= (evap .- cond)
    @views @inbounds @fastmath dydt[vl.masstransferarray[4,:]] .+= (evap .- cond)
    # Vout = sum(evap .- cond)*R*T1/P1
    # @views @inbounds dydt[vl.domain1.indexes[1]:vl.domain1.indexes[2]] .-= Vout*cstot[vl.domain1.indexes[1]:vl.domain1.indexes[2]]
end

function evaluate(vl::VaporLiquidMassTransferInternalInterfaceConstantT,dydt,V1,V2,T1,T2,P1,P2,cstot,p)
    kLAs, kHs = getkLAkHs(vl,T1,T2)
    @views @inbounds @fastmath evap = kLAs.*cstot[vl.masstransferarray[1,:]]*V2
    @views @inbounds @fastmath cond = kLAs./kHs.*cstot[vl.masstransferarray[4,:]]*V1
    @views @inbounds @fastmath dydt[vl.masstransferarray[1,:]] .-= (evap .- cond)
    @views @inbounds @fastmath dydt[vl.masstransferarray[4,:]] .+= (evap .- cond)
    # Vout = sum(evap .- cond)*R*T1/P1
    # @views @inbounds dydt[vl.domain1.indexes[1]:vl.domain1.indexes[2]] .-= Vout*cstot[vl.domain1.indexes[1]:vl.domain1.indexes[2]]
end
export evaluate

struct ReactiveInternalInterfaceConstantTPhi{J,N,B,B2,B3,C,C2,Q<:AbstractReaction} <: AbstractReactiveInternalInterface
    domain1::J
    domain2::N
    reactions::Array{Q,1}
    rxnarray::B
    stoichmatrix::C
    Nrp1::C2
    Nrp2::C2
    kfs::B2
    krevs::B3
    T::Float64
    A::Float64
    parameterindexes::Array{Int64,1}
    domaininds::Array{Int64,1}
    p::Array{Float64,1}
    reversibility::Array{Bool,1}
end
function ReactiveInternalInterfaceConstantTPhi(domain1,domain2,reactions,T,A,phi=0.0)
    @assert domain1.T == domain2.T 
    reactions = upgradekinetics(reactions,domain1,domain2)
    rxnarray = getinterfacereactioninds(domain1,domain2,reactions)
    kfs = getkf.(reactions,nothing,T,0.0,0.0,Ref([]),A,phi)
    Kc = getKc.(reactions,domain1.phase,domain2.phase,Ref(domain1.Gs),Ref(domain2.Gs),T,phi)
    krevs = kfs./Kc
    M,Nrp1,Nrp2 = getstoichmatrix(domain1,domain2,reactions)
    reversibility = Array{Bool,1}(getfield.(reactions,:reversible))
    if isa(reactions,Vector{Any})
        reactions = convert(Vector{ElementaryReaction},reactions)
    end
    if isa(kfs,Vector{Any})
        kfs = convert(Vector{Float64},kfs)
    end
    return ReactiveInternalInterfaceConstantTPhi(domain1,domain2,reactions,
            rxnarray,M,Nrp1,Nrp2,kfs,krevs,T,A,[1,length(reactions)],
            [0,1],kfs[1:end],reversibility),kfs[1:end]
end
export ReactiveInternalInterfaceConstantTPhi

function getkfskrevs(ri::ReactiveInternalInterfaceConstantTPhi,T1,T2,phi1,phi2,Gs1,Gs2,cstot)
    return ri.kfs,ri.krevs
end

function evaluate(ri::ReactiveInternalInterfaceConstantTPhi,dydt,domains,T1,T2,phi1,phi2,Gs1,Gs2,cstot,p::W) where {W<:DiffEqBase.NullParameters}
    addreactionratecontributions!(dydt,ri.rxnarray,cstot,ri.kfs,ri.krevs,ri.A)
end

function evaluate(ri::ReactiveInternalInterfaceConstantTPhi,dydt,domains,T1,T2,phi1,phi2,Gs1,Gs2,cstot,p)
    if p[ri.parameterindexes[1]:ri.parameterindexes[2]] == ri.kfs
        kfs = ri.kfs
    else 
        kfs = p[ri.parameterindexes[1]:ri.parameterindexes[2]]
    end
    if length(Gs1) == 0 || length(Gs2) == 0 || (all(Gs1 .== ri.domain1.Gs) && all(Gs2 .== ri.domain2.Gs))
        krevs = ri.krevs
    else
        Kc = getKcs(ri,T1,Gs1,Gs2)
        krevs = kfs./Kc
    end
    addreactionratecontributions!(dydt,ri.rxnarray,cstot,kfs,krevs,ri.A)
end
export evaluate

"""
construct the stochiometric matrix for the reactions crossing both domains and the reaction molecule # change
"""
function getstoichmatrix(domain1,domain2,rxns)
    M = spzeros(length(rxns),length(domain1.phase.species)+length(domain2.phase.species))
    Nrp1 = zeros(length(rxns))
    Nrp2 = zeros(length(rxns))
    N1 = length(domain1.phase.species)
    spcs1 = domain1.phase.species
    spcs2 = domain2.phase.species
    for (i,rxn) in enumerate(rxns)
        Nrp1[i] = Float64(length([x for x in rxn.products if x in spcs1]) - length([x for x in rxn.reactants if x in spcs1]))
        Nrp2[i] = Float64(length([x for x in rxn.products if x in spcs2]) - length([x for x in rxn.reactants if x in spcs2]))
        for (j,r) in enumerate(rxn.reactants)
            isfirst = true
            ind = findfirst(isequal(r),domain1.phase.species)
            if ind === nothing
                isfirst = false
                ind = findfirst(isequal(r),domain2.phase.species)
            end
            M[i,isfirst ? ind : ind+N1] += 1
        end
        for (j,r) in enumerate(rxn.products)
            isfirst = true
            ind = findfirst(isequal(r),domain1.phase.species)
            if ind === nothing
                isfirst = false
                ind = findfirst(isequal(r),domain2.phase.species)
            end
            M[i,isfirst ? ind : ind+N1] -= 1
        end
    end
    return M,Nrp1,Nrp2
end

function getinterfacereactioninds(domain1,domain2,reactions)
    indices = zeros(Int64,(6,length(reactions)))
    N1 = length(domain1.phase.species)
    for (i,rxn) in enumerate(reactions)
        for (j,r) in enumerate(rxn.reactants)
            isfirst = true
            ind = findfirst(isequal(r),domain1.phase.species)
            if ind === nothing
                isfirst = false
                ind = findfirst(isequal(r),domain2.phase.species)
            end
            indices[j,i] = isfirst ? ind : ind+N1
        end
        for (j,r) in enumerate(rxn.products)
            isfirst = true
            ind = findfirst(isequal(r),domain1.phase.species)
            if ind === nothing
                isfirst = false
                ind = findfirst(isequal(r),domain2.phase.species)
            end
            indices[j+3,i] = isfirst ? ind : ind+N1
        end
    end
    return indices
end

function getinterfacediffusioninds(domain1,domain2,diffusivespcnames)
    indices = zeros(Int64,(6,length(diffusivespcnames)))
    N1 = length(domain1.phase.species)
    spcnames1 = getfield.(domain1.phase.species,:name)
    spcnames2 = getfield.(domain2.phase.species,:name) 
    for (i,name) in enumerate(diffusivespcnames)
        ind1 = findfirst(isequal(name),spcnames1)
        ind2 = findfirst(isequal(name),spcnames2)
        indices[1,i] = ind1
        indices[4,i] = ind2+N1
    end
    return indices
end

function getinterfacemasstransferinds(domain1,domain2,masstransferspcnames)
    indices = zeros(Int64,(6,length(masstransferspcnames)))
    spcnames1 = getfield.(domain1.phase.species,:name)
    spcnames2 = getfield.(domain2.phase.species,:name) 
    for (i,name) in enumerate(masstransferspcnames)
        ind1 = findfirst(isequal(name),spcnames1)
        ind2 = findfirst(isequal(name),spcnames2)
        indices[1,i] = domain2.indexes[1]-1+ind2
        indices[4,i] = domain1.indexes[1]-1+ind1
    end
    return indices
end

function upgradekinetics(rxns,domain1,domain2)
    domain1surf = hasproperty(domain1.phase,:sitedensity)
    domain2surf = hasproperty(domain2.phase,:sitedensity)
    @assert !(domain1surf && domain2surf)
    if domain1surf
        surfdomain = domain1
    elseif domain2surf
        surfdomain = domain2
    end
    newrxns = Array{ElementaryReaction,1}(undef,length(rxns))
    for (i,rxn) in enumerate(rxns)
        if isa(rxn.kinetics,StickingCoefficient)
            spc = [spc for spc in rxn.reactants if !(spc in surfdomain.phase.species)]
            @assert length(spc) == 1
            kin = stickingcoefficient2arrhenius(rxn.kinetics,surfdomain.phase.sitedensity,length(rxn.reactants)-1,spc[1].molecularweight)
            newrxns[i] = ElementaryReaction(index=rxn.index,reactants=rxn.reactants,reactantinds=rxn.reactantinds,products=rxn.products,
                productinds=rxn.productinds,kinetics=kin,radicalchange=rxn.radicalchange,reversible=rxn.reversible,pairs=rxn.pairs)
        else
            newrxns[i] = rxn
        end
    end
    return [rxn for rxn in newrxns]
end

function stickingcoefficient2arrhenius(sc,sitedensity,N,mw;Tmin=300.0,Tmax=2000.0)
    mass = mw/Na
    ksc(T) = sc(T)/sitedensity^N*sqrt(kB*T/(2.0*pi*mass))
    Ts = Array{Float64,1}(Tmin:10:Tmax);
    kscvals = ksc.(Ts)
    k(T,p) = log.(abs(p[1]).*T.^p[2].*exp.(-p[3]./(R.*T)))
    p0 = [sc.A/sitedensity*sqrt(kB*1000.0/(2.0*pi*mass)),0.5,sc.Ea]
    fit = curve_fit(k,Ts,log.(kscvals),p0;x_tol=1e-18)
    @assert fit.converged
    p = fit.param
    p[1] = abs(p[1])
    return Arrhenius(;A=p[1],n=p[2],Ea=p[3])
end

struct Inlet{Q<:Real,S,V<:AbstractArray,U<:Real,X<:Real,FF<:Function} <: AbstractBoundaryInterface
    domain::S
    y::V
    F::FF
    T::U
    P::X
    H::Q
end

function Inlet(domain::V,conddict::Dict{X1,X},F::FF) where {V,X1,X,B<:Real,FF<:Function}
    y = makespcsvector(domain.phase,conddict)
    T = conddict["T"]
    P = conddict["P"]
    yout = y./sum(y)
    H = dot(getEnthalpy.(getfield.(domain.phase.species,:thermo),T),yout)
    return Inlet(domain,yout,F,T,P,H)
end

export Inlet

struct Outlet{V,FF<:Function} <: AbstractBoundaryInterface
    domain::V
    F::FF
end
export Outlet

struct TPDependentOutlet{V,FF<:Real,PP<:Real,TT<:Real} <: AbstractBoundaryInterface
    domain::V
    F::FF
    P::PP
    T::TT
end
export TPDependentOutlet

struct ConstantVaporVolumeOutlet{V} <: AbstractBoundaryInterface
    domain::V
end
export ConstantVaporVolumeOutlet

"""
kLAkHCondensationEvaporationWithReservoir adds evaporation and condensation to
(1) a liquid phase domain with a constant composition vapor resevoir, where molefractions, P, and T need to be specified, or
(2) a gas phase domain with a constant composition liquid resevoir, where number of moles, V, and T need to be specified.
kLA and kH are used to model cond/evap. 
kLA is liquid volumetric mass transfer coefficient with unit 1/s , and kH is Henry's law constant.
"""

struct kLAkHCondensationEvaporationWithReservoir{S,V1<:AbstractArray,V2<:Real,V3<:Real,V4<:AbstractArray,V5<:Real,V6<:AbstractArray,V7<:AbstractArray,V8<:Real} <: AbstractBoundaryInterface
    domain::S
    molefractions::V1
    T::V2
    P::V3
    V::V8
    cs::V4
    H::V5
    kLAs::V6
    kHs::V7
end

function kLAkHCondensationEvaporationWithReservoir(domain::D,conddict::Dict{X1,X}) where {D,X1,X}
    y = makespcsvector(domain.phase,conddict)
    molefractions = y./sum(y)
    T = conddict["T"]
    H = dot(getEnthalpy.(getfield.(domain.phase.species,:thermo),T),molefractions)
    kLAs = [T -> kLA(T=T) for kLA in getfield.(domain.phase.species,:liquidvolumetricmasstransfercoefficient)]
    kHs = [T -> kH(T=T) for kH in getfield.(domain.phase.species,:henrylawconstant)]

    if isa(domain.phase,IdealDiluteSolution)
        if !haskey(conddict,"P")
            @error "P needs to be specified for the vapor resevoir over the liquid phase domain"
        end
        P = conddict["P"]
        return kLAkHCondensationEvaporationWithReservoir(domain,molefractions,T,P,0.0,Array{Float64,1}(),H,kLAs,kHs)
    elseif isa(domain.phase,IdealGas)
        if !haskey(conddict,"V")
            @error "V needs to be specified for the liquid resevoir under the gas phase domain"
        end
        V = conddict["V"]
        cs = y./V
        return kLAkHCondensationEvaporationWithReservoir(domain,Array{Float64,1}(),T,1e8,V,cs,H,kLAs,kHs)
    end
end

export kLAkHCondensationEvaporationWithReservoir

struct VolumetricFlowRateOutlet{V,F1<:Function} <: AbstractBoundaryInterface
    domain::V
    Vout::F1
end
export VolumetricFlowRateOutlet
