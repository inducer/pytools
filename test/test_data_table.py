# data from Wikipedia "join" article


def get_dept_table():
    from pytools.datatable import DataTable
    dept_table = DataTable(["id", "name"])
    dept_table.insert_row((31, "Sales"))
    dept_table.insert_row((33, "Engineering"))
    dept_table.insert_row((34, "Clerical"))
    dept_table.insert_row((35, "Marketing"))
    return dept_table


def get_employee_table():
    from pytools.datatable import DataTable
    employee_table = DataTable(["lastname", "dept"])
    employee_table.insert_row(("Rafferty", 31))
    employee_table.insert_row(("Jones", 33))
    employee_table.insert_row(("Jasper", 36))
    employee_table.insert_row(("Steinberg", 33))
    employee_table.insert_row(("Robinson", 34))
    employee_table.insert_row(("Smith", 34))
    return employee_table


def test_len():
    et = get_employee_table()
    assert len(et) == 6


def test_iter():
    et = get_employee_table()

    count = 0
    for row in et:
        count += 1
        assert len(row) == 2

    assert count == 6


def test_insert_and_get():
    et = get_employee_table()
    et.insert(dept=33, lastname="Kloeckner")
    assert et.get(lastname="Kloeckner").dept == 33


def test_filtered():
    et = get_employee_table()
    assert len(et.filtered(dept=33)) == 2
    assert len(et.filtered(dept=34)) == 2


def test_sort():
    et = get_employee_table()
    et.sort(["lastname"])
    assert et.column_data("dept") == [36, 33, 31, 34, 34, 33]


def test_aggregate():
    et = get_employee_table()
    et.sort(["dept"])
    agg = et.aggregated(["dept"], "lastname", ",".join)
    assert len(agg) == 4
    for dept, lastnames in agg:
        lastnames = lastnames.split(",")
        for lastname in lastnames:
            assert et.get(lastname=lastname).dept == dept


def test_aggregate_2():
    from pytools.datatable import DataTable
    tbl = DataTable(["step", "value"], list(zip(list(range(20)), list(range(20)))))
    agg = tbl.aggregated(["step"], "value", max)
    assert agg.column_data("step") == list(range(20))
    assert agg.column_data("value") == list(range(20))


def test_join():
    et = get_employee_table()
    dt = get_dept_table()

    et.sort(["dept"])
    dt.sort(["id"])

    inner_joined = et.join("dept", "id", dt)
    assert len(inner_joined) == len(et)-1
    for dept, lastname, deptname in inner_joined:
        dept_id = et.get(lastname=lastname).dept
        assert dept_id == dept
        assert dt.get(id=dept_id).name == deptname

    outer_joined = et.join("dept", "id", dt, outer=True)
    assert len(outer_joined) == len(et)+1
