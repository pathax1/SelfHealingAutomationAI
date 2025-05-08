from behave import given, when, then
from Common_Functions.CommonFunctions import click, type_text, verify_element
from Common_Functions.CommonFunctions import add_screenshot_to_report

@given('the user has launched the ZARA site and reject cookies')
def step_reject_cookies(context):
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path,
                             context.config.userdata["testcase"])
    click(context.driver, 'Reject_Cookies')

@when('the user clicks on the login links')
def step_click_login_link(context):
    click(context.driver, 'login_Menu_Link')
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path,context.config.userdata["testcase"])
    click(context.driver, 'Log_in_Button')

@when('the user enters an invalid email and password')
def step_enter_invalid_credentials(context):
    #so the syntax is testdata column , what if that column doesn't exist some kind of failsafe
    email = context.testdata.get("Email", "default@example.com")
    pwd = context.testdata.get("Passcode", "defaultPass")
    type_text(context.driver, 'Email', email)
    type_text(context.driver, 'Password', pwd)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path,context.config.userdata["testcase"])

@when('the user clicks on the login button')
def step_click_login_button(context):
    click(context.driver, 'Login_Button')

@then('an error message should be displayed')
def step_verify_error_message(context):
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path,context.config.userdata["testcase"])
    verify_element(context.driver, 'login_error_message')


