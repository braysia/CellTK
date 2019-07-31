from utils.traces import TracesController
from utils.pairwise import one_to_one_assignment, one_to_two_assignment, angle_assignment
import numpy as np
from utils.traces import construct_traces_based_on_next, convert_traces_to_storage, label_traces
np.random.seed(0)


def gap_closing(cells, DISPLACEMENT=100, MASSTHRES=0.15, maxgap=4):
    '''
    Connect cells between non-consecutive frames if it meets criteria.
    maxgap (int): the maximum frames allowed to connect two cells.
    '''
    traces = construct_traces_based_on_next(cells)
    trhandler = TracesController(traces)

    # make sure not to have a cell as both disappered and appeared cells
    store_singleframe = []
    for trace in trhandler.traces[:]:
        if len(trace) < 2:
            trhandler.traces.remove(trace)
            store_singleframe.append(trace)
    dist = trhandler.pairwise_dist()
    massdiff = trhandler.pairwise_mass()
    framediff = trhandler.pairwise_frame()

    withinarea = dist < DISPLACEMENT
    inmass = abs(massdiff) < MASSTHRES
    inframe = (framediff > 1) * (framediff <= maxgap)
    withinarea_inframe = withinarea * inframe * inmass
    # CHECK: distance as a fine cost
    withinarea_inframe = one_to_one_assignment(withinarea_inframe, dist)
    if withinarea_inframe.any():
        disapp_idx, app_idx = np.where(withinarea_inframe)

        dis_cells = trhandler.disappeared()
        app_cells = trhandler.appeared()
        for disi, appi in zip(disapp_idx, app_idx):
            dis_cell, app_cell = dis_cells[disi], app_cells[appi]
            dis_cell.nxt = app_cell

            # You can simply reconstruct the trace, but here to reduce the calculation,
            # connect them explicitly.
            dis_trace = [i for i in trhandler.traces if dis_cell in i][0]
            app_trace = [i for i in trhandler.traces if app_cell in i][0]
            dis_trace.extend(trhandler.traces.pop(trhandler.traces.index(app_trace)))

    traces = traces + store_singleframe
    traces = label_traces(trhandler.traces)
    return convert_traces_to_storage(traces)


def cut_short_traces(cells, minframe=4):
    '''

    '''
    if max([i.frame for i in cells]) < minframe:
        print "minframe set to the maximum"
        minframe = max([i.frame for i in cells])

    '''handle division'''
    def list_parent_daughters(cells):
        cc = [(i.parent, i.label) for i in cells if i.parent is not None]
        parents = set([i[0] for i in cc])
        parents = list(parents)
        store = []
        for pt in parents:
            daughters = [i[1] for i in cc if i[0] == pt]
            store.append([pt] + daughters)
        return store
    pdsets = list_parent_daughters(cells)
    traces = construct_traces_based_on_next(cells)
    for pdset in pdsets:
        p0 = traces.pop([n for n, i in enumerate(traces) if pdset[0] == i[-1].label][0])
        d0 = traces.pop([n for n, i in enumerate(traces) if pdset[1] == i[0].label][0])
        d1 = traces.pop([n for n, i in enumerate(traces) if pdset[2] == i[0].label][0])
        traces.append(p0 + d0)
        traces.append(p0 + d1)
    ''' Calculate the largest frame differences so it will go well with gap closing'''
    store = []
    for trace in traces:
        frames = [i.frame for i in trace]
        if max(frames) - min(frames) >= minframe:
            store.append(trace)
    return convert_traces_to_storage(store)


def detect_division(cells, METHOD='mass', DISPLACEMENT=50, maxgap=4, DIVISIONMASSERR=0.15, DOT_THRES=-0.7, DIST_THRES=0.35, ANG_WEIGHT = 0.5):
    '''
    Params:
        METHOD: 'mass' or 'angle'
            mass matches daughter cells based on total mass of daughter cells closest to parent cell total mass
            angle matches daughter cells based on the angle between their vectors and the parent cell. DOT_THRES is used to determine cutoff
            in angle method, with multiple candidate pairs, mass is used to determine correct pair, based on ANG_WEIGHT
            Note: angle method will fail to capture dividing cells that are also moving significantly during division
        DISPLACEMENT: Max distance from parent to potential daughter cells
        maxgap: Max amount of frames from division event to matched daughter cell
        DIVISIONMASSERR: Threshold error for difference in total mass between parents and daughters
        DOT_THRES: Range -1 to 1.Threshold of the cosine of the angle between the vectors of the daughter cells to the parent
        DIST_THRES: Range 0 to 1. Threshold is measured in fraction from center point of two daughters. 0 is parent exactly in middle. 1 is parent at edge
        ANG_WEIGHT: Range 0 to 1. 0=all mass, 1=all angle. Balance between angle cost and mass cost when resolving multiple possible daughter pairs. 
    '''
    traces = construct_traces_based_on_next(cells)
    trhandler = TracesController(traces)
    store_singleframe = []
    for trace in trhandler.traces[:]:
        if len(trace) < 2:
            trhandler.traces.remove(trace)
            store_singleframe.append(trace)

    dist = trhandler.pairwise_dist()
    massdiff = trhandler.pairwise_mass()
    framediff = trhandler.pairwise_frame()
    half_massdiff = massdiff + 0.5

    withinarea = dist < DISPLACEMENT
    inframe = (framediff <= maxgap) * (framediff >= 1)
    halfmass = abs(half_massdiff) < DIVISIONMASSERR

    withinarea_inframe_halfmass = withinarea * inframe * halfmass

    dis_cells = trhandler.disappeared()
    app_cells = trhandler.appeared()

    if METHOD=='mass':
        # CHECK: now based on distance.
        par_dau = one_to_two_assignment(withinarea_inframe_halfmass, half_massdiff)
        # CHECK: If only one daughter is found ignore it.
        par_dau[par_dau.sum(axis=1) == 1] = False
    elif METHOD=='angle':
        par_xy, dau_xy = trhandler.pairwise_xy()
        par_dau = angle_assignment(withinarea_inframe_halfmass, dau_xy, par_xy, dot_thres=DOT_THRES, dist_thres=DIST_THRES, mass_cost=half_massdiff, weight=ANG_WEIGHT)
        
        #remove double assigned daughters
        #could be optional in the future
        par_dau[:, par_dau.sum(axis=0) > 1] = False
        par_dau[par_dau.sum(axis=1) == 1] = False

    if par_dau.any():
        disapp_idx, app_idx = np.where(par_dau)

        for disi, appi in zip(disapp_idx, app_idx):
            dis_cell = dis_cells[disi]
            app_cell = app_cells[appi]
            app_cell.parent = dis_cell.label
            # dis_cell.nxt = app_cell
    return convert_traces_to_storage(trhandler.traces + store_singleframe)
